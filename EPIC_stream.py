import os
import json
import time
import faiss
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not installed. Plotting functions will be disabled.")

class StreamSetup:
    """
    Stream-based evaluation setup that processes documents in batches (e.g., 2000 docs)
    and evaluates performance at each checkpoint with dynamic preference management.
    
    Uses incremental FAISS updates with metadata management for efficiency.
    """
    
    def __init__(self, utils, batch_size=2000, skip_evaluation=False, stream_seed=None):
        self.utils = utils
        self.method = utils.method
        self.device = utils.device
        self.output_dir = utils.output_dir
        self.emb_model_name = utils.emb_model_name
        self.doc_mode = utils.doc_mode
        self.batch_size = batch_size  # Documents per batch (default: 2000)
        self.skip_evaluation = skip_evaluation  # Skip evaluation during checkpoint
        self.stream_seed = stream_seed  # Seed for preference events (for reproducibility)
        
        # Stream state - using metadata-based management
        self.chunk_metadata = []  # [{id, text, instruction, relevant_preferences, active}]
        self.next_chunk_id = 0
        self.faiss_index = None  # Single FAISS index with IDMap
        self.embedding_dim = None
        self.checkpoint_results = []
        
        # Preference management
        self.active_preferences = []  # Currently active preferences
        self.inactive_preferences = []  # Removed/inactive preferences
        self.preference_history = []  # Log of preference changes
        self.preference_to_chunks = {}  # {preference: set(chunk_ids)} mapping
        
        # Stream metadata
        self.stream_meta = {
            "start_time": None,
            "checkpoints": [],
            "preference_events": [],
            "total_docs_processed": 0,
            "total_chunks_indexed": 0,
            "active_chunks": 0
        }
    
    def initialize_stream(self, persona_index, all_chunks, all_embeddings):
        """Initialize stream with persona data and full document corpus"""
        self.persona_index = persona_index
        self.all_chunks = all_chunks
        self.all_embeddings = all_embeddings
        
        
        # Load persona data (needed for all methods for generation/evaluation)
        # Standard method doesn't use preferences for indexing/retrieval, but still needs persona for evaluation
        self.persona = self.utils.load_persona_data(persona_index)
        self.active_preferences = [block["preference"] for block in self.persona["preference_blocks"]]
        
        # Initialize preference embeddings (skip for standard method in indexing/retrieval)
        if self.method != "standard":
            self._compute_preference_embeddings()
        else:
            # For standard: don't compute preference embeddings (not used in indexing/retrieval)
            self.preference_embeddings = None
        
        # Reset stream state
        self.chunk_metadata = []
        self.next_chunk_id = 0
        self.checkpoint_results = []
        if self.method == "EPIC_inst":
            self.preference_to_chunks = {pref: set() for pref in self.active_preferences}
        else:
            # For standard and cosine: no preference_to_chunks mapping needed
            self.preference_to_chunks = {}
        
        # Initialize FAISS index with IDMap for incremental updates
        self.embedding_dim = all_embeddings.shape[1] if len(all_embeddings) > 0 else 768
        self._init_faiss_index(self.embedding_dim)
        
        # Reset metadata
        self.stream_meta["start_time"] = datetime.now().isoformat()
        self.stream_meta["total_docs_processed"] = 0
        self.stream_meta["total_chunks_indexed"] = 0
        self.stream_meta["active_chunks"] = 0
        
        print(f"✅ Stream initialized for persona {persona_index}")
        print(f"   Total documents available: {len(all_chunks)}")
        print(f"   Active preferences: {len(self.active_preferences)}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   FAISS index: IndexIDMap (incremental updates enabled)")
    
    def _init_faiss_index(self, dim):
        """Initialize FAISS index with IDMap for incremental updates and potential deletions"""
        base_index = faiss.IndexFlatIP(dim)
        self.faiss_index = faiss.IndexIDMap(base_index)
        print(f"✅ Initialized FAISS IndexIDMap with dimension {dim}")
    
    def _compute_preference_embeddings(self):
        """Compute embeddings for active preferences"""
        if self.active_preferences:
            self.preference_embeddings = self.utils.embed_texts_mp(self.active_preferences)
            self.preference_embeddings = self.preference_embeddings / np.linalg.norm(
                self.preference_embeddings, axis=1, keepdims=True
            )
        else:
            self.preference_embeddings = None
    
    def _llm_filter_chunk(self, chunk, relevant_preferences):
        """
        LLM filtering to decide Keep/Discard for a chunk using utils.process_chunk_rand_prefs
        
        Args:
            chunk: Document chunk text
            relevant_preferences: List of relevant preference texts (from cosine filtering)
        
        Returns:
            tuple: (keep: bool, filtered_preferences: list, reason: str)
        """
        try:
            # Load filtering prompts for EPIC_inst
            system_prompt = self.utils.load_prompt_template(self.utils.filtering_inst_system)
            user_prompt = self.utils.load_prompt_template(self.utils.filtering_inst_user)
            
            # Use utils.process_chunk_rand_prefs
            result = self.utils.process_chunk_rand_prefs(
                idx=0,  # Not used for single chunk
                chunk_text=chunk,
                preference_text="",  # Will be formatted inside
                prompt_template=user_prompt,
                prompt_template_system=system_prompt,
                preference_list=relevant_preferences
            )
            
            if result.get("status") != "success":
                return False, [], result.get("reason", "Processing failed")
            
            decision = result.get("decision", "Filter")
            reason = result.get("reason", "")
            preferences = result.get("relevant_preference", [])
            
            keep = decision == "Keep"
            return keep, preferences if preferences else [], reason
            
        except Exception as e:
            print(f"LLM filtering failed: {e}")
            return False, [], f"Error: {str(e)}"
    
    def add_preference(self, preference_text, source_persona_index=None):
        """
        Add a new preference (from another persona or custom)
        
        Args:
            preference_text: The preference text to add
            source_persona_index: Source persona index (for logging)
        """
        if preference_text in self.active_preferences:
            print(f"⚠️ Preference already active: {preference_text[:50]}...")
            return False
        
        # If it was inactive, reactivate it
        reactivated_count = 0
        if preference_text in self.inactive_preferences:
            self.inactive_preferences.remove(preference_text)
            
            # Reactivate chunks that belong to this preference (only for EPIC_inst)
            if self.method == "EPIC_inst":
                if preference_text in self.preference_to_chunks:
                    chunk_ids = self.preference_to_chunks[preference_text]
                    for chunk_id in chunk_ids:
                        if chunk_id < len(self.chunk_metadata):
                            meta = self.chunk_metadata[chunk_id]
                            if not meta.get("active", True):
                                meta["active"] = True
                                meta["reactivated_at_docs"] = self.stream_meta["total_docs_processed"]
                                reactivated_count += 1
        
        self.active_preferences.append(preference_text)
        # Update preference embeddings (skip for standard method)
        if self.method != "standard":
            self._compute_preference_embeddings()
        
        # Initialize preference_to_chunks if new preference (only for EPIC_inst)
        if self.method == "EPIC_inst":
            if preference_text not in self.preference_to_chunks:
                self.preference_to_chunks[preference_text] = set()
        
        # Update active chunks count
        if self.method == "EPIC_inst":
            active_count = sum(1 for m in self.chunk_metadata if m.get("active", True))
        else:
            # For standard and cosine: all chunks (no active concept)
            active_count = len(self.chunk_metadata)
        self.stream_meta["active_chunks"] = active_count
        
        # Log the event
        event = {
            "type": "add",
            "preference": preference_text,
            "source_persona": source_persona_index,
            "timestamp": datetime.now().isoformat(),
            "docs_processed": self.stream_meta["total_docs_processed"],
            "chunks_reactivated": reactivated_count
        }
        self.preference_history.append(event)
        self.stream_meta["preference_events"].append(event)
        
        print(f"✅ Added preference: {preference_text[:50]}...")
        if reactivated_count > 0:
            print(f"   Reactivated {reactivated_count} chunks")
        print(f"   Active preferences: {len(self.active_preferences)}")
        print(f"   Active chunks: {active_count}")
        return True
    
    def remove_preference(self, preference_text):
        """
        Remove/deactivate a preference and mark related documents as inactive
        
        Args:
            preference_text: The preference text to remove
        """
        if preference_text not in self.active_preferences:
            print(f"⚠️ Preference not active: {preference_text[:50]}...")
            return False
        
        self.active_preferences.remove(preference_text)
        self.inactive_preferences.append(preference_text)
        # Update preference embeddings (skip for standard method)
        if self.method != "standard":
            self._compute_preference_embeddings()
        
        # Deactivate chunks that ONLY belong to this preference (only for EPIC_inst)
        deactivated_count = 0
        if self.method == "EPIC_inst":
            if preference_text in self.preference_to_chunks:
                chunk_ids = self.preference_to_chunks[preference_text]
                for chunk_id in chunk_ids:
                    if chunk_id < len(self.chunk_metadata):
                        meta = self.chunk_metadata[chunk_id]
                        # Check if chunk has other active preferences
                        other_active_prefs = [p for p in meta["relevant_preferences"] 
                                             if p in self.active_preferences and p != preference_text]
                        if not other_active_prefs:
                            # No other active preferences, deactivate this chunk
                            meta["active"] = False
                            meta["deactivated_at_docs"] = self.stream_meta["total_docs_processed"]
                            meta["deactivated_reason"] = f"Preference removed: {preference_text[:50]}..."
                            deactivated_count += 1
        
        # Update active chunks count
        if self.method == "EPIC_inst":
            active_count = sum(1 for m in self.chunk_metadata if m.get("active", True))
        else:
            # For standard and cosine: all chunks (no active concept)
            active_count = len(self.chunk_metadata)
        self.stream_meta["active_chunks"] = active_count
        
        # Log the event
        event = {
            "type": "remove",
            "preference": preference_text,
            "timestamp": datetime.now().isoformat(),
            "docs_processed": self.stream_meta["total_docs_processed"],
            "chunks_deactivated": deactivated_count
        }
        self.preference_history.append(event)
        self.stream_meta["preference_events"].append(event)
        
        print(f"❌ Removed preference: {preference_text[:50]}...")
        print(f"   Deactivated {deactivated_count} chunks")
        print(f"   Active preferences: {len(self.active_preferences)}")
        print(f"   Active chunks: {active_count}")
        return True
    
    def get_random_preference_from_other_persona(self, exclude_persona_index=None):
        """
        Get a random preference from another persona
        
        Args:
            exclude_persona_index: Persona index to exclude (usually current persona)
        
        Returns:
            tuple: (preference_text, source_persona_index)
        """
        if exclude_persona_index is None:
            exclude_persona_index = self.persona_index
        
        # Get all available persona indices based on dataset
        if self.utils.dataset_name == "PrefWiki":
            all_indices = list(range(57))
        elif self.utils.dataset_name == "PrefRQ":
            all_indices = list(range(90))
        elif self.utils.dataset_name == "PrefELI5":
            all_indices = list(range(73))
        elif self.utils.dataset_name == "PrefEval":
            all_indices = list(range(57))
        else:
            all_indices = list(range(10))
        
        # Remove current persona
        available_indices = [i for i in all_indices if i != exclude_persona_index]
        
        if not available_indices:
            print("⚠️ No other personas available")
            return None, None
        
        # Random select a persona
        source_persona_index = random.choice(available_indices)
        source_persona = self.utils.load_persona_data(source_persona_index)
        
        # Get all preferences from that persona
        source_preferences = [block["preference"] for block in source_persona["preference_blocks"]]
        
        # Filter out already active preferences
        available_prefs = [p for p in source_preferences if p not in self.active_preferences]
        
        if not available_prefs:
            # Try another persona recursively (with limit)
            return self.get_random_preference_from_other_persona(exclude_persona_index)
        
        selected_preference = random.choice(available_prefs)
        return selected_preference, source_persona_index
    
    def search_active_chunks(self, query_emb, top_k=5):
        """
        Search FAISS index and return only active chunks (for EPIC_inst only)
        
        Args:
            query_emb: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            list: Top-k active chunks
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        # Search more than top_k to account for inactive chunks
        search_k = min(top_k * 3, self.faiss_index.ntotal)
        D, I = self.faiss_index.search(query_emb.astype(np.float32), search_k)
        
        results = []
        for idx in I[0]:
            if idx >= 0 and idx < len(self.chunk_metadata):
                meta = self.chunk_metadata[idx]
                # Only filter by active for EPIC_inst
                if self.method == "EPIC_inst":
                    if meta.get("active", True):
                        results.append(meta["text"])
                        if len(results) >= top_k:
                            break
                else:
                    # For standard and cosine: no active filtering
                    results.append(meta["text"])
                    if len(results) >= top_k:
                        break
        
        return results
    
    def get_active_chunks(self):
        """Get list of all active chunks (for EPIC_inst) or all chunks (for standard/cosine)"""
        if self.method == "EPIC_inst":
            return [m["text"] for m in self.chunk_metadata if m.get("active", True)]
        else:
            # For standard and cosine: return all chunks (no active filtering)
            return [m["text"] for m in self.chunk_metadata]
    
    def run_checkpoint_evaluation(self, checkpoint_id, method_dir):
        """
        Run evaluation at current checkpoint
        
        Args:
            checkpoint_id: Identifier for this checkpoint
            method_dir: Directory to save results
        
        Returns:
            dict: Evaluation results with 5 metrics
        """
        print(f"\n🔍 Running checkpoint evaluation #{checkpoint_id}...")
        
        # Get existing FAISS index (no rebuild needed)
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("⚠️ No embeddings in index")
            return None
        
        # Save current chunks for generation
        checkpoint_dir = os.path.join(method_dir, f"checkpoint_{checkpoint_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save chunks (with instructions if available)
        # For standard and cosine: save all chunks
        # For EPIC_inst: save only active chunks
        kept_file = os.path.join(checkpoint_dir, "kept.jsonl")
        active_chunks = []
        with open(kept_file, 'w', encoding='utf-8') as f:
            for meta in self.chunk_metadata:
                # For standard and cosine: save all chunks (no active filtering)
                # For EPIC_inst: save only active chunks
                if self.method in ["standard", "cosine"] or meta.get("active", True):
                    item = {"text": meta["text"], "id": meta["id"]}
                    if meta.get("instruction"):
                        item["instruction"] = meta["instruction"]
                    item["relevant_preferences"] = meta["relevant_preferences"]
                    if meta.get("processing_time") is not None:
                        item["processing_time"] = meta["processing_time"]
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    active_chunks.append(meta["text"])
        
        # Run generation for all queries
        generation_results = self._run_generation(checkpoint_dir, active_chunks)
        
        # Run evaluation on generated results (skip if flag is set)
        if self.skip_evaluation:
            print(f"⏭️ Skipping evaluation (skip_evaluation=True)")
            evaluation_stats = {}
        else:
            evaluation_stats = self._run_evaluation(generation_results, checkpoint_dir)
        
        # Calculate metrics
        total = len(generation_results) if generation_results else 1
        # For standard and cosine: all indexed chunks are "active"
        if self.method == "EPIC_inst":
            active_count = len(active_chunks)
        else:
            active_count = len(self.chunk_metadata)
        
        # Calculate processing time statistics
        processing_times = [m.get("processing_time", 0) for m in self.chunk_metadata if m.get("processing_time") is not None]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        max_processing_time = np.max(processing_times) if processing_times else 0
        min_processing_time = np.min(processing_times) if processing_times else 0
        total_processing_time = np.sum(processing_times) if processing_times else 0
        
        # Calculate memory usage for this checkpoint
        memory_mb = self._calculate_checkpoint_memory(checkpoint_dir)
        
        metrics = {
            "checkpoint_id": checkpoint_id,
            "docs_processed": self.stream_meta["total_docs_processed"],
            "total_indexed": len(self.chunk_metadata),
            "active_chunks": active_count,
            "active_preferences": len(self.active_preferences),
            "unhelpful": evaluation_stats.get("error_unhelpful", 0),
            "inconsistent": evaluation_stats.get("error_inconsistent", 0),
            "hallucination_of_preference_violation": evaluation_stats.get("hallucination_of_preference_violation", 0),
            "preference_unaware_violation": evaluation_stats.get("preference_unaware_violation", 0),
            "preference_following_accuracy": round((evaluation_stats.get("preference_adherence_accuracy", 0) / total) * 100, 2) if not self.skip_evaluation else 0,
            "avg_processing_time_per_chunk": round(avg_processing_time, 4),
            "max_processing_time_per_chunk": round(max_processing_time, 4),
            "min_processing_time_per_chunk": round(min_processing_time, 4),
            "total_processing_time": round(total_processing_time, 2),
            "memory_mb": round(memory_mb, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        self.checkpoint_results.append(metrics)
        self.stream_meta["checkpoints"].append(metrics)
        
        # Save checkpoint results
        self._save_checkpoint_results(checkpoint_dir, metrics)
        
        # Save to data/indexing directory (for consistency with non-stream mode)
        self._save_to_data_indexing(checkpoint_id)
        
        print(f"✅ Checkpoint #{checkpoint_id} complete")
        print(f"   Total indexed: {len(self.chunk_metadata)}, Active: {active_count}")
        if not self.skip_evaluation:
            print(f"   Accuracy: {metrics['preference_following_accuracy']}%")
        if metrics.get('avg_processing_time_per_chunk', 0) > 0:
            print(f"   Avg processing time per chunk: {metrics['avg_processing_time_per_chunk']:.4f}s")
            print(f"   Total processing time: {metrics['total_processing_time']:.2f}s")
        
        return metrics
    
    def _run_generation(self, checkpoint_dir, active_chunks=None):
        """Run generation for current checkpoint"""
        generation_prompt = self.utils.load_prompt_template(self.utils.generation_prompt)
        all_results = []
        
        # Get chunks if not provided
        # For standard and cosine: get all chunks
        # For EPIC methods: get only active chunks
        if active_chunks is None:
            if self.method in ["standard", "cosine"]:
                active_chunks = [m["text"] for m in self.chunk_metadata]
            else:
                active_chunks = self.get_active_chunks()
        
        # Process each active preference block
        # Standard method also uses persona for evaluation (same as non-stream mode)
        for block in self.persona["preference_blocks"]:
            preference_text = block["preference"]
            
            # Skip if preference is inactive
            if preference_text not in self.active_preferences:
                continue
            
            queries = block["queries"]  # Use all queries
            
            for query in queries:
                question = query["question"]
                
                try:
                    start_retrieval = time.time()
                    
                    # Use active-aware retrieval
                    retrieved_data = self._retrieve_active_chunks(question, self.utils.top_k)
                    retrieval_time = time.time() - start_retrieval
                    
                    # Handle different return types
                    if isinstance(retrieved_data, dict):
                        # EPIC_inst returns dict with 'texts' and 'instructions'
                        retrieved = retrieved_data.get("texts", [])
                        retrieved_instructions = retrieved_data.get("instructions", [])
                    else:
                        # Standard/cosine returns list of texts
                        retrieved = retrieved_data if retrieved_data else []
                        retrieved_instructions = []
                    
                    if not retrieved:
                        # Fallback to standard retrieval if no results (using utils functions)
                        if self.method in ["standard", "cosine"]:
                            retrieved, retrieval_time = self.utils.retrieve_top_k(
                                question,
                                self.faiss_index,
                                active_chunks
                            )
                            retrieved_instructions = []
                        else:
                            retrieved, retrieval_time = self.utils.retrieve_top_k_wq_cosine(
                                question,
                                self.active_preferences,
                                self.faiss_index,
                                active_chunks
                            )
                            retrieved_instructions = []
                    
                    # Build context with instructions if available (same as EPIC_generation.py)
                    if self.method == "EPIC_inst" and retrieved_instructions:
                        context_parts = []
                        for i, (doc, inst) in enumerate(zip(retrieved, retrieved_instructions)):
                            if inst:
                                context_parts.append(f"Document {i+1}:\nInterpretation Guidance: {inst}\nContent: {doc}")
                            else:
                                context_parts.append(f"Document {i+1}: {doc}")
                        context = "\n\n".join(context_parts)
                    else:
                        context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved)])
                    filled_prompt = generation_prompt.replace("{context}", context).replace("{question}", question)
                    
                    generated_text = self.utils.generate_message_vllm(
                        messages=[{"role": "user", "content": filled_prompt}],
                        system_prompt="You are a helpful assistant for generating responses.",
                        max_tokens=2048
                    )
                    
                    if generated_text:
                        all_results.append({
                            "preference": preference_text,
                            "question": question,
                            "response_to_q": generated_text,
                            "retrieved_docs": retrieved,
                            "retrieval_time": retrieval_time
                        })
                except Exception as e:
                    print(f"Generation error: {e}")
                    continue
        
        # Save generation results
        gen_file = os.path.join(checkpoint_dir, "generation_results.json")
        self.utils.save_json(gen_file, all_results)
        
        return all_results
    
    def _retrieve_active_chunks(self, question, top_k=5):
        """
        Retrieve top-k chunks for a question
        
        Args:
            question: Query question
            top_k: Number of results
        
        Returns:
            list: Retrieved chunk texts
        """
        # Compute query embedding
        query_emb = self.utils.embed_query_mp(question)
        
        # For standard method: use retrieve_top_k (mydata style)
        if self.method == "standard":
            # Use retrieve_top_k without preference weighting
            active_chunks_list = [m["text"] for m in self.chunk_metadata]
            retrieved, _ = self.utils.retrieve_top_k(
                question,
                self.faiss_index,
                active_chunks_list
            )
            return retrieved
        
        # For cosine method: use retrieve_top_k (mydata style - no preference weighting)
        if self.method == "cosine":
            # Use simple retrieval without preference weighting (mydata style)
            active_chunks_list = [m["text"] for m in self.chunk_metadata]
            retrieved, _ = self.utils.retrieve_top_k(
                question,
                self.faiss_index,
                active_chunks_list
            )
            return retrieved
        
        # For EPIC_inst: add preference weighting and filter by top preference (same as EPIC_generation.py)
        if self.method == "EPIC_inst":
            if self.preference_embeddings is not None and len(self.active_preferences) > 0:
                # Find top-1 preference for this query (same as EPIC_generation.py)
                pref_sims = np.dot(self.preference_embeddings, query_emb.T).squeeze()
                max_pref_idx = np.argmax(pref_sims)
                top_pref_text = self.active_preferences[max_pref_idx]
                
                # Augment query with preference (same as EPIC_generation.py - not weighted)
                pref_emb = self.utils.embed_query_mp(top_pref_text)
                query_emb = query_emb + pref_emb
                query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
                
                # Search with larger k to allow for filtering (same as EPIC_generation.py)
                search_k = min(top_k * 5, self.faiss_index.ntotal)
                D, I = self.faiss_index.search(query_emb.astype(np.float32), search_k)
                
                # Filter by top_pref_text in relevant_preferences and active status (same as EPIC_generation.py)
                retrieved = []
                retrieved_instructions = []
                chunk_id_to_meta = {meta["id"]: meta for meta in self.chunk_metadata}
                for chunk_id in I[0]:
                    if chunk_id in chunk_id_to_meta:
                        meta = chunk_id_to_meta[chunk_id]
                        # Check if chunk is active and relevant to top preference
                        # Use relevant_preferences (stream mode) instead of preference_ids (generation mode)
                        if meta.get("active", True) and top_pref_text in meta.get("relevant_preferences", []):
                            retrieved.append(meta["text"])
                            retrieved_instructions.append(meta.get("instruction", ""))
                            if len(retrieved) >= top_k:
                                break
                
                return {"texts": retrieved, "instructions": retrieved_instructions}
        
        # Fallback for other EPIC methods (should not reach here for supported methods)
        return self.search_active_chunks(query_emb, top_k)
    
    def _run_evaluation(self, generation_data, checkpoint_dir):
        """Run evaluation on generated results"""
        if not generation_data:
            return {}
        
        # Load evaluation prompts
        file_dict = {
            "acknow": "check_acknowledge.txt",
            "violate": "check_violation.txt",
            "hallucinate": "check_hallucination.txt",
            "helpful": "check_helpful.txt"
        }
        
        eval_message_texts = {}
        for metric_name, file_name in file_dict.items():
            file_path = os.path.join(self.utils.error_type_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                eval_message_texts[metric_name] = f.read()
        
        system_prompt = "You are a helpful assistant in evaluating an AI assistant's response."
        
        # Evaluate each result
        for task in tqdm(generation_data, desc="Evaluating", leave=False):
            task["evaluation_error_analysis"] = {}
            
            for metric, eval_text in eval_message_texts.items():
                try:
                    filled_text = eval_text
                    if metric == "acknow":
                        filled_text = filled_text.replace("{end_generation}", task["response_to_q"]).replace("{question}", task["question"])
                    elif metric in ["violate", "helpful"]:
                        filled_text = filled_text.replace("{preference}", task["preference"]).replace("{question}", task["question"]).replace("{end_generation}", task["response_to_q"])
                    elif metric == "hallucinate":
                        extracted_pref = task.get("evaluation_error_analysis", {}).get("acknow", {}).get("extract_pref", "")
                        filled_text = filled_text.replace("{preference}", task["preference"]).replace("{assistant_restatement}", extracted_pref)
                    
                    eval_response = self.utils.generate_message_vllm(
                        [{"role": "user", "content": filled_text}],
                        system_prompt
                    )
                    
                    if eval_response:
                        if metric != "acknow":
                            explanation, answer = self.utils.parse_explanation_and_answer(eval_response)
                            task["evaluation_error_analysis"][metric] = {"explanation": explanation, "answer": answer}
                        else:
                            extract_pref, answer = self.utils.parse_preference_and_answer(eval_response)
                            task["evaluation_error_analysis"][metric] = {"answer": answer, "extract_pref": extract_pref}
                except Exception as e:
                    print(f"Evaluation error for {metric}: {e}")
        
        # Calculate stats
        stats = {
            "acknowledgement": 0,
            "hallucination": 0,
            "violation": 0,
            "error_unhelpful": 0,
            "error_inconsistent": 0,
            "hallucination_of_preference_violation": 0,
            "preference_unaware_violation": 0,
            "preference_adherence_accuracy": 0,
        }
        
        for entry in generation_data:
            if "evaluation_error_analysis" not in entry:
                continue
            
            error_types = entry["evaluation_error_analysis"]
            is_acknowledgement = "yes" in error_types.get("acknow", {}).get("answer", "").lower()
            is_hallucination = is_acknowledgement and "yes" in error_types.get("hallucinate", {}).get("answer", "").lower()
            is_violation = "yes" in error_types.get("violate", {}).get("answer", "").lower()
            is_unhelpful = "no" in error_types.get("helpful", {}).get("answer", "").lower()
            
            is_inconsistent = is_acknowledgement and not is_hallucination and is_violation and not is_unhelpful
            is_hallucination_of_preference_violation = is_acknowledgement and is_hallucination and is_violation and not is_unhelpful
            is_preference_unaware_violation = not is_acknowledgement and is_violation and not is_unhelpful
            
            preference_following_accuracy = not any([
                is_inconsistent, is_hallucination_of_preference_violation, 
                is_preference_unaware_violation, is_unhelpful
            ])
            
            stats["acknowledgement"] += is_acknowledgement
            stats["hallucination"] += is_hallucination
            stats["violation"] += is_violation
            stats["error_unhelpful"] += is_unhelpful
            stats["error_inconsistent"] += is_inconsistent
            stats["hallucination_of_preference_violation"] += is_hallucination_of_preference_violation
            stats["preference_unaware_violation"] += is_preference_unaware_violation
            stats["preference_adherence_accuracy"] += preference_following_accuracy
        
        # Save evaluation results
        eval_file = os.path.join(checkpoint_dir, "evaluation_results.json")
        self.utils.save_json(eval_file, generation_data)
        
        return stats
    
    def _save_checkpoint_results(self, checkpoint_dir, metrics):
        """Save checkpoint results and metadata"""
        # Save metrics
        metrics_file = os.path.join(checkpoint_dir, "metrics.json")
        self.utils.save_json(metrics_file, metrics)
        
        # Save current preference state
        pref_state_file = os.path.join(checkpoint_dir, "preference_state.json")
        self.utils.save_json(pref_state_file, {
            "active_preferences": self.active_preferences,
            "inactive_preferences": self.inactive_preferences,
            "preference_history": self.preference_history,
            "preference_to_chunks_count": {k: len(v) for k, v in self.preference_to_chunks.items()}
        })
        
        # Save chunk metadata summary
        chunk_summary_file = os.path.join(checkpoint_dir, "chunk_summary.json")
        if self.method == "EPIC_inst":
            active_count = sum(1 for m in self.chunk_metadata if m.get("active", True))
            inactive_count = len(self.chunk_metadata) - active_count
        else:
            # For standard and cosine: no active concept
            active_count = len(self.chunk_metadata)
            inactive_count = 0
        self.utils.save_json(chunk_summary_file, {
            "total_indexed": len(self.chunk_metadata),
            "active": active_count,
            "inactive": inactive_count
        })
    
    def _calculate_checkpoint_memory(self, checkpoint_dir):
        """
        Calculate memory usage for current checkpoint
        
        Args:
            checkpoint_dir: Checkpoint directory path
            
        Returns:
            float: Total memory in MB
        """
        total_memory_mb = 0.0
        
        # 1. Calculate kept.jsonl size
        kept_file = os.path.join(checkpoint_dir, "kept.jsonl")
        if os.path.exists(kept_file):
            total_memory_mb += os.path.getsize(kept_file) / (1024 * 1024)
        
        # 2. Calculate data/indexing files size
        # Build data directory path for stream mode
        root_dir = self.utils.root_dir
        doc_mode_stream = f"{self.doc_mode}_stream"
        
        # Determine method suffix
        if self.utils.llm_model_name == "openai/gpt-oss-20b":
            llm_suffix = "_oss"
        elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            llm_suffix = "_qwen"
        else:
            llm_suffix = ""
        
        # Build data_dir path for stream mode
        if self.utils.dataset_name == "PrefWiki":
            method_name = f"{self.method}_prefwiki{llm_suffix}"
        elif self.utils.dataset_name == "PrefELI5":
            method_name = f"{self.method}_prefeli5{llm_suffix}"
        elif self.utils.dataset_name == "PrefRQ":
            method_name = f"{self.method}_rq{llm_suffix}"
        elif self.utils.dataset_name == "PrefEval":
            method_name = f"{self.method}_prefeval{llm_suffix}"
        else:
            method_name = f"{self.method}_prefwiki{llm_suffix}"
        
        data_dir = os.path.join(root_dir, "indexing", doc_mode_stream, method_name, str(self.persona_index))
        
        # Calculate embeddings and index file sizes
        model_name_clean = self.emb_model_name.replace('/', '_')
        embeddings_file = os.path.join(data_dir, f"embeddings_{model_name_clean}.npy")
        index_file = os.path.join(data_dir, f"index_{model_name_clean}.faiss")
        
        if os.path.exists(embeddings_file):
            total_memory_mb += os.path.getsize(embeddings_file) / (1024 * 1024)
        
        if os.path.exists(index_file):
            total_memory_mb += os.path.getsize(index_file) / (1024 * 1024)
        
        # 3. Calculate preference embeddings if exists
        # Map dataset name to preference embeddings file naming
        if self.utils.dataset_name == "PrefWiki":
            pref_emb_dataset_name = "prefwiki"
        elif self.utils.dataset_name == "PrefELI5":
            pref_emb_dataset_name = "prefeli5"
        elif self.utils.dataset_name == "PrefRQ":
            pref_emb_dataset_name = "rq"
        elif self.utils.dataset_name == "PrefEval":
            pref_emb_dataset_name = "prefeval"
        else:
            pref_emb_dataset_name = "prefwiki"
        
        pref_emb_file = os.path.join(root_dir, "indexing", f"preference_embeddings_{self.persona_index}_{pref_emb_dataset_name}_mp.npy")
        if os.path.exists(pref_emb_file):
            total_memory_mb += os.path.getsize(pref_emb_file) / (1024 * 1024)
        
        return total_memory_mb
    
    def _save_to_data_indexing(self, checkpoint_id):
        """
        Save FAISS index and embeddings to data/indexing directory
        Stream mode uses {doc_mode}_stream to distinguish from non-stream mode
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return
        
        # Get data directory path for stream mode (use {doc_mode}_stream)
        # Build path similar to utils.data_dir but with _stream suffix
        root_dir = self.utils.root_dir
        doc_mode_stream = f"{self.doc_mode}_stream"
        
        # Determine method suffix
        if self.utils.llm_model_name == "openai/gpt-oss-20b":
            llm_suffix = "_oss"
        elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            llm_suffix = "_qwen"
        else:
            llm_suffix = ""
        
        # Build data_dir path for stream mode
        if self.utils.dataset_name == "PrefWiki":
            method_name = f"{self.method}_prefwiki{llm_suffix}"
        elif self.utils.dataset_name == "PrefELI5":
            method_name = f"{self.method}_prefeli5{llm_suffix}"
        elif self.utils.dataset_name == "PrefRQ":
            method_name = f"{self.method}_rq{llm_suffix}"
        elif self.utils.dataset_name == "PrefEval":
            method_name = f"{self.method}_prefeval{llm_suffix}"
        else:
            method_name = f"{self.method}_prefwiki{llm_suffix}"
        
        data_dir = os.path.join(root_dir, "indexing", doc_mode_stream, method_name, str(self.persona_index))
        os.makedirs(data_dir, exist_ok=True)
        
        model_name_clean = self.emb_model_name.replace('/', '_')
        
        # Save FAISS index
        index_file = os.path.join(data_dir, f"index_{model_name_clean}.faiss")
        faiss.write_index(self.faiss_index, index_file)
        
        # Extract and save embeddings from chunk_metadata
        # Collect embeddings in order of chunk_id
        if self.chunk_metadata:
            # Sort by chunk_id to ensure correct order
            sorted_metadata = sorted(self.chunk_metadata, key=lambda x: x["id"])
            embeddings_list = []
            
            for meta in sorted_metadata:
                if "embedding" in meta:
                    embeddings_list.append(meta["embedding"])
            
            if embeddings_list:
                embeddings_array = np.array(embeddings_list).astype(np.float32)
                embeddings_file = os.path.join(data_dir, f"embeddings_{model_name_clean}.npy")
                np.save(embeddings_file, embeddings_array)
                print(f"💾 Saved embeddings to {embeddings_file} ({len(embeddings_list)} vectors)")
        
        print(f"💾 Saved FAISS index to {index_file}")
    
    def run_stream(self, method_dir, preference_events=None, checkpoint_interval=None):
        """
        Run the full stream processing pipeline (document-by-document)
        Supports resume from last processed document index
        
        Args:
            method_dir: Output directory for results
            preference_events: Optional list of preference change events
                Format: [{"type": "add"|"remove", "at_docs": int, "preference": str|None}]
                If preference is None for "add", will get random from other persona
            checkpoint_interval: Number of documents between checkpoints (default: batch_size)
        
        Returns:
            dict: Stream results with all checkpoint metrics
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Stream Processing (Document-by-Document)")
        print(f"{'='*60}")
        
        # Try to find existing stream directory for resume
        stream_dir = self._find_or_create_stream_dir(method_dir)
        resume_info = self._load_resume_info(stream_dir)
        
        total_docs = len(self.all_chunks)
        checkpoint_interval = checkpoint_interval or self.batch_size
        num_checkpoints = (total_docs + checkpoint_interval - 1) // checkpoint_interval
        
        # Resume from last processed index
        start_doc_idx = 0
        checkpoint_id = 0
        docs_since_checkpoint = 0
        event_idx = 0
        
        if resume_info:
            print(f"📂 Resuming from previous run...")
            start_doc_idx = resume_info.get("last_processed_idx", 0) + 1
            checkpoint_id = resume_info.get("last_checkpoint_id", 0)
            docs_since_checkpoint = resume_info.get("docs_since_checkpoint", 0)
            event_idx = resume_info.get("event_idx", 0)
            
            # Restore FAISS index and metadata
            self._restore_state(stream_dir)
            
            print(f"   Last processed document: {start_doc_idx - 1}")
            print(f"   Resuming from document: {start_doc_idx}")
            print(f"   Last checkpoint: {checkpoint_id}")
        else:
            print(f"🆕 Starting new stream run...")
            os.makedirs(stream_dir, exist_ok=True)
        
        print(f"Total documents: {total_docs}")
        print(f"Processing: Document-by-document")
        print(f"Checkpoint interval: Every {checkpoint_interval} documents")
        print(f"Expected checkpoints: {num_checkpoints}")
        
        # Sort preference events by docs threshold
        if preference_events:
            preference_events = sorted(preference_events, key=lambda x: x.get("at_docs", 0))
        else:
            preference_events = []
        
        # Process documents one by one
        for doc_idx in tqdm(range(start_doc_idx, total_docs), desc="Processing documents", initial=start_doc_idx, total=total_docs):
            # Check for preference events (all methods need this for evaluation)
            while event_idx < len(preference_events):
                event = preference_events[event_idx]
                if event.get("at_docs", 0) <= self.stream_meta["total_docs_processed"]:
                    # Pass stream_seed for reproducibility across methods
                    self._handle_preference_event(event, event_seed=self.stream_seed)
                    event_idx += 1
                else:
                    break
            
            # Process single document
            self._process_single_document(doc_idx)
            docs_since_checkpoint += 1
            
            # Save progress periodically (every 100 documents)
            if doc_idx % 100 == 0:
                self._save_progress(stream_dir, doc_idx, checkpoint_id, docs_since_checkpoint, event_idx)
            
            # Run checkpoint at intervals
            if docs_since_checkpoint >= checkpoint_interval:
                checkpoint_id += 1
                print(f"\n--- Checkpoint {checkpoint_id} at {self.stream_meta['total_docs_processed']} docs ---")
                self.run_checkpoint_evaluation(checkpoint_id, stream_dir)
                docs_since_checkpoint = 0
                # Save progress after checkpoint
                self._save_progress(stream_dir, doc_idx, checkpoint_id, docs_since_checkpoint, event_idx)
        
        # Final checkpoint if there are remaining docs
        if docs_since_checkpoint > 0:
            checkpoint_id += 1
            print(f"\n--- Final Checkpoint {checkpoint_id} at {self.stream_meta['total_docs_processed']} docs ---")
            self.run_checkpoint_evaluation(checkpoint_id, stream_dir)
        
        # Save final stream results
        self._save_stream_results(stream_dir)
        
        # Clean up progress file on successful completion
        progress_file = os.path.join(stream_dir, "progress.json")
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        print(f"\n{'='*60}")
        print(f"✅ Stream Processing Complete")
        print(f"{'='*60}")
        print(f"Total checkpoints: {len(self.checkpoint_results)}")
        print(f"Results saved to: {stream_dir}")
        
        return {
            "stream_dir": stream_dir,
            "checkpoints": self.checkpoint_results,
            "metadata": self.stream_meta
        }
    
    def _process_single_document(self, doc_idx):
        """
        Process a single document and add to index if relevant
        
        Args:
            doc_idx: Index of document in all_chunks
            
        Flow:
            1. Cosine filtering - check if chunk is similar to any active preference
            2. LLM filtering - ask LLM to decide Keep/Discard (for EPIC_inst)
            3. If Keep -> Instruction generation (for EPIC_inst)
            4. Add to FAISS index
        """
        process_start_time = time.time()
        
        chunk = self.all_chunks[doc_idx]
        emb = self.all_embeddings[doc_idx]
        
        # Normalize embedding
        emb_norm = emb / np.linalg.norm(emb)
        
        # Update processed count first
        self.stream_meta["total_docs_processed"] += 1
        
        # ============================================
        # Standard method: Save all chunks without filtering
        # ============================================
        if self.method == "standard":
            processing_time = time.time() - process_start_time
            # Save all chunks with empty preferences (no preference filtering)
            self._add_single_chunk_to_index(chunk, emb_norm, [], processing_time=processing_time)
            return
        
        # ============================================
        # Step 1: Cosine Filtering
        # ============================================
        cosine_passed = False
        cosine_relevant_prefs = []
        
        if self.preference_embeddings is not None and len(self.active_preferences) > 0:
            # Compute similarity with all active preferences
            sims = np.dot(self.preference_embeddings, emb_norm)
            above_threshold = sims > self.utils.threshold
            
            if np.any(above_threshold):
                cosine_passed = True
                relevant_pref_indices = np.where(above_threshold)[0]
                cosine_relevant_prefs = [self.active_preferences[idx] for idx in relevant_pref_indices]
        else:
            # No preferences, keep all
            cosine_passed = True
            cosine_relevant_prefs = self.active_preferences[:]
        
        # For cosine method: save if cosine filtering passed, skip LLM filtering
        if self.method == "cosine":
            if cosine_passed:
                processing_time = time.time() - process_start_time
                self._add_single_chunk_to_index(chunk, emb_norm, cosine_relevant_prefs, processing_time=processing_time)
            return
        
        # If cosine filtering failed, skip this document
        if not cosine_passed:
            return
        
        # ============================================
        # Step 2: LLM Filtering (Keep/Discard)
        # ============================================
        # For EPIC_inst: need LLM filtering
        if self.method == "EPIC_inst":
            llm_keep, llm_prefs, reason = self._llm_filter_chunk(chunk, cosine_relevant_prefs)
            
            if not llm_keep:
                return  # Discard
            
            final_prefs = llm_prefs if llm_prefs else cosine_relevant_prefs
            
            # Generate instruction using utils functions
            entry = {
                "chunk": chunk,
                "relevant_preference": final_prefs,
                "reason": reason if reason else "Matched by LLM filtering"
            }
            inst_prompt_user = self.utils.load_prompt_template(self.utils.inst_user)
            inst_prompt_system = self.utils.load_prompt_template(self.utils.inst_system)
            inst_result = self.utils.inst_single(entry, inst_prompt_user, inst_prompt_system)
            instruction = inst_result.get("instruction", "")
        else:
            # For cosine method, no LLM filtering
            final_prefs = cosine_relevant_prefs
            instruction = None
        
        # ============================================
        # Step 3: Add to FAISS Index
        # ============================================
        processing_time = time.time() - process_start_time
        
        if self.method == "EPIC_inst" and instruction:
            # For EPIC_inst: use instruction embedding for FAISS
            instruction_emb = self.utils.embed_query_mp(instruction)
            instruction_emb = instruction_emb / np.linalg.norm(instruction_emb, axis=1, keepdims=True)
            self._add_single_chunk_to_index(chunk, instruction_emb.squeeze(0), final_prefs, instruction, processing_time)
        else:
            # For other methods: use chunk embedding
            self._add_single_chunk_to_index(chunk, emb_norm, final_prefs, None, processing_time)
    
    def _add_single_chunk_to_index(self, chunk, embedding, preferences, instruction=None, processing_time=None):
        """
        Add a single chunk to the FAISS index
        
        Args:
            chunk: Chunk text
            embedding: Normalized embedding vector
            preferences: List of relevant preferences
            instruction: Optional instruction text (for EPIC_inst)
            processing_time: Time taken to process this chunk (in seconds)
        """
        chunk_id = self.next_chunk_id
        
        # Add to FAISS
        emb_array = np.array([embedding]).astype(np.float32)
        ids = np.array([chunk_id], dtype=np.int64)
        self.faiss_index.add_with_ids(emb_array, ids)
        
        # Add metadata (field names match EPIC_indexing.py)
        metadata = {
            "id": chunk_id,
            "text": chunk,  # "text" to match indexing
            "relevant_preferences": preferences,  # "relevant_preferences" to match indexing
            "added_at_docs": self.stream_meta["total_docs_processed"],
            "embedding": embedding  # Store embedding for saving to data/indexing
        }
        
        # Only add "active" field for EPIC_inst (standard and cosine don't use active filtering)
        if self.method == "EPIC_inst":
            metadata["active"] = True
        
        # Add method-specific fields
        if instruction:
            metadata["instruction"] = instruction
        if processing_time is not None:
            metadata["processing_time"] = processing_time
        
        self.chunk_metadata.append(metadata)
        
        # Update preference_to_chunks mapping (only for EPIC_inst)
        if self.method == "EPIC_inst":
            for pref in preferences:
                if pref not in self.preference_to_chunks:
                    self.preference_to_chunks[pref] = set()
                self.preference_to_chunks[pref].add(chunk_id)
        
        self.next_chunk_id += 1
        self.stream_meta["total_chunks_indexed"] = len(self.chunk_metadata)
        
        # Update active chunks count
        if self.method == "EPIC_inst":
            self.stream_meta["active_chunks"] = sum(1 for m in self.chunk_metadata if m.get("active", True))
        else:
            # For standard and cosine: all chunks (no active concept)
            self.stream_meta["active_chunks"] = len(self.chunk_metadata)
    
    def _handle_preference_event(self, event, event_seed=None):
        """
        Handle a preference change event
        
        Args:
            event: Preference event dict
            event_seed: Optional seed for random operations (for reproducibility)
        """
        event_type = event.get("type")
        preference = event.get("preference")
        
        # Set seed for this event if provided (for reproducibility across methods)
        if event_seed is not None:
            random.seed(event_seed)
            np.random.seed(event_seed)
        
        if event_type == "add":
            if preference is None:
                # Get random preference from other persona
                preference, source_persona = self.get_random_preference_from_other_persona()
                if preference:
                    self.add_preference(preference, source_persona)
            else:
                self.add_preference(preference)
        
        elif event_type == "remove":
            if preference is None:
                # Remove random active preference
                if self.active_preferences:
                    preference = random.choice(self.active_preferences)
                    self.remove_preference(preference)
            else:
                self.remove_preference(preference)
    
    def _find_or_create_stream_dir(self, method_dir):
        """Find existing stream directory or create new one"""
        # Look for existing stream directories
        if os.path.exists(method_dir):
            stream_dirs = [d for d in os.listdir(method_dir) if d.startswith("stream_") and os.path.isdir(os.path.join(method_dir, d))]
            if stream_dirs:
                # Get most recent stream directory
                stream_dirs.sort(reverse=True)
                latest_dir = os.path.join(method_dir, stream_dirs[0])
                progress_file = os.path.join(latest_dir, "progress.json")
                if os.path.exists(progress_file):
                    print(f"📂 Found existing stream directory: {latest_dir}")
                    return latest_dir
        
        # Create new stream directory
        stream_dir = os.path.join(method_dir, f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        return stream_dir
    
    def _load_resume_info(self, stream_dir):
        """Load resume information from progress file"""
        progress_file = os.path.join(stream_dir, "progress.json")
        if os.path.exists(progress_file):
            try:
                return self.utils.load_json(progress_file)
            except Exception as e:
                print(f"⚠️ Failed to load progress file: {e}")
                return None
        return None
    
    def _save_progress(self, stream_dir, last_processed_idx, checkpoint_id, docs_since_checkpoint, event_idx):
        """Save current progress for resume"""
        progress_file = os.path.join(stream_dir, "progress.json")
        progress_info = {
            "last_processed_idx": last_processed_idx,
            "last_checkpoint_id": checkpoint_id,
            "docs_since_checkpoint": docs_since_checkpoint,
            "event_idx": event_idx,
            "total_docs_processed": self.stream_meta["total_docs_processed"],
            "total_chunks_indexed": len(self.chunk_metadata),
            "active_chunks": sum(1 for m in self.chunk_metadata if m.get("active", True)) if self.method == "EPIC_inst" else len(self.chunk_metadata),
            "timestamp": datetime.now().isoformat()
        }
        self.utils.save_json(progress_file, progress_info)
    
    def _restore_state(self, stream_dir):
        """Restore metadata from last checkpoint"""
        try:
            # Find the latest checkpoint directory
            checkpoint_dirs = [d for d in os.listdir(stream_dir) if d.startswith("checkpoint_") and os.path.isdir(os.path.join(stream_dir, d))]
            if not checkpoint_dirs:
                print("⚠️ No checkpoints found, starting fresh...")
                return
            
            # Get the latest checkpoint
            checkpoint_dirs.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0, reverse=True)
            latest_checkpoint_dir = os.path.join(stream_dir, checkpoint_dirs[0])
            
            # Load preference state from checkpoint
            pref_state_file = os.path.join(latest_checkpoint_dir, "preference_state.json")
            if os.path.exists(pref_state_file):
                pref_state = self.utils.load_json(pref_state_file)
                self.active_preferences = pref_state.get("active_preferences", [])
                self.inactive_preferences = pref_state.get("inactive_preferences", [])
                self.preference_history = pref_state.get("preference_history", [])
                
                # Recompute preference embeddings
                self._compute_preference_embeddings()
                
                print(f"✅ Restored preference state from checkpoint {checkpoint_dirs[0]}")
                print(f"   Active preferences: {len(self.active_preferences)}")
            
            # Load chunk metadata from kept.jsonl in checkpoint
            kept_file = os.path.join(latest_checkpoint_dir, "kept.jsonl")
            if os.path.exists(kept_file):
                restored_chunks = 0
                with open(kept_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            chunk_id = item.get("id", len(self.chunk_metadata))
                            metadata = {
                                "id": chunk_id,
                                "text": item["text"],
                                "instruction": item.get("instruction"),
                                "relevant_preferences": item.get("relevant_preferences", []),
                                "active": True,
                                "added_at_docs": self.stream_meta.get("total_docs_processed", 0)
                            }
                            if item.get("processing_time") is not None:
                                metadata["processing_time"] = item["processing_time"]
                            self.chunk_metadata.append(metadata)
                            
                            # Update preference_to_chunks mapping
                            for pref in metadata["relevant_preferences"]:
                                if pref not in self.preference_to_chunks:
                                    self.preference_to_chunks[pref] = set()
                                self.preference_to_chunks[pref].add(chunk_id)
                            
                            restored_chunks += 1
                
                self.next_chunk_id = max([m["id"] for m in self.chunk_metadata], default=0) + 1
                print(f"✅ Restored {restored_chunks} chunks from checkpoint")
            
            # Rebuild FAISS index from saved embeddings
            if self.chunk_metadata:
                # Try to load embeddings from indexing directory
                root_dir = self.utils.root_dir
                doc_mode_stream = f"{self.doc_mode}_stream"
                
                # Determine method suffix
                if self.utils.llm_model_name == "openai/gpt-oss-20b":
                    llm_suffix = "_oss"
                elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
                    llm_suffix = "_qwen"
                else:
                    llm_suffix = ""
                
                # Build data_dir path for stream mode
                if self.utils.dataset_name == "PrefWiki":
                    method_name = f"{self.method}_prefwiki{llm_suffix}"
                elif self.utils.dataset_name == "PrefELI5":
                    method_name = f"{self.method}_prefeli5{llm_suffix}"
                elif self.utils.dataset_name == "PrefRQ":
                    method_name = f"{self.method}_rq{llm_suffix}"
                elif self.utils.dataset_name == "PrefEval":
                    method_name = f"{self.method}_prefeval{llm_suffix}"
                else:
                    method_name = f"{self.method}_prefwiki{llm_suffix}"
                
                data_dir = os.path.join(root_dir, "indexing", doc_mode_stream, method_name, str(self.persona_index))
                model_name_clean = self.emb_model_name.replace('/', '_')
                embeddings_file = os.path.join(data_dir, f"embeddings_{model_name_clean}.npy")
                index_file = os.path.join(data_dir, f"index_{model_name_clean}.faiss")
                
                # Try to load FAISS index directly
                if os.path.exists(index_file):
                    try:
                        self.faiss_index = faiss.read_index(index_file)
                        print(f"✅ Restored FAISS index from {index_file} ({self.faiss_index.ntotal} vectors)")
                        
                        # Load embeddings and update metadata
                        if os.path.exists(embeddings_file):
                            embeddings_array = np.load(embeddings_file)
                            # Match embeddings with metadata by chunk_id
                            sorted_metadata = sorted(self.chunk_metadata, key=lambda x: x["id"])
                            for i, meta in enumerate(sorted_metadata):
                                if i < len(embeddings_array):
                                    meta["embedding"] = embeddings_array[i]
                    except Exception as e:
                        print(f"⚠️ Failed to load FAISS index: {e}")
                        # Fallback: initialize empty index
                        if self.faiss_index is None:
                            self._init_faiss_index(self.embedding_dim)
                else:
                    # No saved index, initialize empty
                    if self.faiss_index is None:
                        self._init_faiss_index(self.embedding_dim)
                        print(f"⚠️ No saved FAISS index found, initialized empty index")
                
        except Exception as e:
            print(f"⚠️ Failed to restore state: {e}")
            print(f"   Starting fresh...")
            # Initialize fresh state if restore fails
            if self.faiss_index is None:
                self._init_faiss_index(self.embedding_dim)
    
    def _save_stream_results(self, stream_dir):
        """Save final stream results and summary"""
        # Save all checkpoint results
        results_file = os.path.join(stream_dir, "all_checkpoints.json")
        self.utils.save_json(results_file, self.checkpoint_results)
        
        # Save stream metadata
        meta_file = os.path.join(stream_dir, "stream_metadata.json")
        self.utils.save_json(meta_file, self.stream_meta)
        
        # Save preference history
        history_file = os.path.join(stream_dir, "preference_history.json")
        self.utils.save_json(history_file, self.preference_history)
        
        # Generate summary CSV for plotting
        self._generate_summary_csv(stream_dir)
    
    def _generate_summary_csv(self, stream_dir):
        """Generate CSV summary for easy plotting"""
        csv_file = os.path.join(stream_dir, "checkpoint_summary.csv")
        
        fieldnames = [
            "checkpoint_id",
            "docs_processed",
            "total_indexed",
            "active_chunks",
            "active_preferences",
            "unhelpful",
            "inconsistent", 
            "hallucination_of_preference_violation",
            "preference_unaware_violation",
            "preference_following_accuracy",
            "avg_processing_time_per_chunk",
            "max_processing_time_per_chunk",
            "min_processing_time_per_chunk",
            "total_processing_time",
            "memory_mb",
            "timestamp"
        ]
        
        # Write header
        with open(csv_file, 'w') as f:
            f.write(",".join(fieldnames) + "\n")
        
        # Write each checkpoint
        for checkpoint in self.checkpoint_results:
            row = {field: checkpoint.get(field, "") for field in fieldnames}
            self.utils.save_csv(csv_file, fieldnames, row, write_header=False)
        
        print(f"📊 Summary CSV saved to: {csv_file}")
        
        # Generate plot if matplotlib is available
        if HAS_MATPLOTLIB:
            self.plot_stream_results(stream_dir)
    
    def plot_stream_results(self, stream_dir):
        """
        Generate visualization of stream results with 5 evaluation axes
        Shows preference events (add/remove) as vertical lines
        """
        if not HAS_MATPLOTLIB:
            print("⚠️ matplotlib not installed. Cannot generate plots.")
            return
        
        if not self.checkpoint_results:
            print("⚠️ No checkpoint results to plot.")
            return
        
        # Extract data for plotting
        docs_processed = [cp["docs_processed"] for cp in self.checkpoint_results]
        
        metrics = {
            "Unhelpful": [cp["unhelpful"] for cp in self.checkpoint_results],
            "Inconsistent": [cp["inconsistent"] for cp in self.checkpoint_results],
            "Hallucination Violation": [cp["hallucination_of_preference_violation"] for cp in self.checkpoint_results],
            "Unaware Violation": [cp["preference_unaware_violation"] for cp in self.checkpoint_results],
            "Accuracy (%)": [cp["preference_following_accuracy"] for cp in self.checkpoint_results]
        }
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71']
        
        # Plot 1: Error metrics (counts)
        ax1 = axes[0]
        for i, (metric_name, values) in enumerate(list(metrics.items())[:-1]):
            ax1.plot(docs_processed, values, marker='o', label=metric_name, 
                    color=colors[i], linewidth=2, markersize=6)
        
        ax1.set_ylabel('Error Count', fontsize=12)
        ax1.set_title('Stream Evaluation: Error Metrics Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2 = axes[1]
        ax2.plot(docs_processed, metrics["Accuracy (%)"], marker='s', 
                color=colors[4], linewidth=2, markersize=8, label='Preference Following Accuracy')
        ax2.fill_between(docs_processed, metrics["Accuracy (%)"], alpha=0.3, color=colors[4])
        
        ax2.set_xlabel('Documents Processed', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Stream Evaluation: Preference Following Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add preference events as vertical lines
        add_events = [e for e in self.preference_history if e["type"] == "add"]
        remove_events = [e for e in self.preference_history if e["type"] == "remove"]
        
        for ax in axes:
            for event in add_events:
                ax.axvline(x=event["docs_processed"], color='green', linestyle='--', 
                          alpha=0.7, linewidth=1.5)
            for event in remove_events:
                ax.axvline(x=event["docs_processed"], color='red', linestyle='--', 
                          alpha=0.7, linewidth=1.5)
        
        # Add legend for events
        if add_events or remove_events:
            add_patch = mpatches.Patch(color='green', alpha=0.7, label='Preference Added')
            remove_patch = mpatches.Patch(color='red', alpha=0.7, label='Preference Removed')
            patches = []
            if add_events:
                patches.append(add_patch)
            if remove_events:
                patches.append(remove_patch)
            if patches:
                axes[0].legend(handles=list(axes[0].get_legend_handles_labels()[0]) + patches, 
                              loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(stream_dir, "stream_evaluation_plot.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to: {plot_file}")
        
        # Also create a combined metrics plot
        self._plot_combined_metrics(stream_dir, docs_processed, metrics)
    
    def _plot_combined_metrics(self, stream_dir, docs_processed, metrics):
        """Create a combined plot with all 5 metrics normalized"""
        if not HAS_MATPLOTLIB:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71']
        markers = ['o', 's', '^', 'D', 'v']
        
        # Normalize error counts to percentage for comparison
        max_errors = max(max(metrics["Unhelpful"]) if metrics["Unhelpful"] else 1,
                        max(metrics["Inconsistent"]) if metrics["Inconsistent"] else 1,
                        max(metrics["Hallucination Violation"]) if metrics["Hallucination Violation"] else 1,
                        max(metrics["Unaware Violation"]) if metrics["Unaware Violation"] else 1,
                        1)
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            if metric_name == "Accuracy (%)":
                plot_values = values
            else:
                # Invert and normalize: fewer errors = higher score
                plot_values = [100 - (v / max_errors * 100) if max_errors > 0 else 100 for v in values]
                metric_name = f"No {metric_name} (%)"
            
            ax.plot(docs_processed, plot_values, marker=markers[i], label=metric_name,
                   color=colors[i], linewidth=2, markersize=6)
        
        # Add preference events
        for event in self.preference_history:
            color = 'green' if event["type"] == "add" else 'red'
            ax.axvline(x=event["docs_processed"], color=color, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Documents Processed', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Stream Evaluation: All Metrics (Higher is Better)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        plot_file = os.path.join(stream_dir, "stream_combined_plot.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Combined plot saved to: {plot_file}")
    
    def reprocess_with_new_preference_events(self, stream_dir, new_preference_events, 
                                             start_from_checkpoint=None):
        """
        Reprocess an existing stream with new preference events
        
        This function allows you to re-run evaluation with different preference
        event timings without reprocessing all documents.
        
        Args:
            stream_dir: Existing stream directory
            new_preference_events: New preference events to apply
            start_from_checkpoint: Checkpoint ID to start from (None = from beginning)
        
        Returns:
            dict: Updated stream results
        """
        print(f"\n{'='*60}")
        print(f"🔄 Reprocessing Stream with New Preference Events")
        print(f"{'='*60}")
        
        # Load existing stream metadata
        meta_file = os.path.join(stream_dir, "stream_metadata.json")
        if not os.path.exists(meta_file):
            print(f"❌ Stream metadata not found: {meta_file}")
            return None
        
        stream_meta = self.utils.load_json(meta_file)
        original_preference_history = stream_meta.get("preference_events", [])
        
        print(f"📂 Original preference events: {len(original_preference_history)}")
        print(f"📂 New preference events: {len(new_preference_events)}")
        
        # Find checkpoints
        checkpoint_dirs = [d for d in os.listdir(stream_dir) 
                         if d.startswith("checkpoint_") and os.path.isdir(os.path.join(stream_dir, d))]
        checkpoint_dirs.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
        
        if not checkpoint_dirs:
            print("❌ No checkpoints found to reprocess")
            return None
        
        # Determine starting checkpoint
        if start_from_checkpoint is None:
            start_checkpoint_id = 1
        else:
            start_checkpoint_id = start_from_checkpoint
        
        print(f"🔄 Starting reprocessing from checkpoint {start_checkpoint_id}")
        
        # Restore state from the checkpoint before start_checkpoint_id
        if start_checkpoint_id > 1:
            prev_checkpoint_dir = os.path.join(stream_dir, f"checkpoint_{start_checkpoint_id - 1}")
            if os.path.exists(prev_checkpoint_dir):
                self._restore_state_from_checkpoint(prev_checkpoint_dir)
                print(f"✅ Restored state from checkpoint {start_checkpoint_id - 1}")
        
        # Update preference history with new events
        self.preference_history = []
        self.stream_meta["preference_events"] = []
        
        # Apply new preference events up to current point
        current_docs = self.stream_meta.get("total_docs_processed", 0)
        for event in sorted(new_preference_events, key=lambda x: x.get("at_docs", 0)):
            if event.get("at_docs", 0) <= current_docs:
                self._handle_preference_event(event)
        
        # Reprocess checkpoints
        updated_checkpoint_results = []
        for checkpoint_dir_name in checkpoint_dirs:
            checkpoint_id = int(checkpoint_dir_name.split("_")[1])
            if checkpoint_id < start_checkpoint_id:
                # Keep original results
                metrics_file = os.path.join(stream_dir, checkpoint_dir_name, "metrics.json")
                if os.path.exists(metrics_file):
                    original_metrics = self.utils.load_json(metrics_file)
                    updated_checkpoint_results.append(original_metrics)
                continue
            
            checkpoint_dir = os.path.join(stream_dir, checkpoint_dir_name)
            
            # Apply preference events that occurred before this checkpoint
            checkpoint_meta = self.utils.load_json(os.path.join(checkpoint_dir, "metrics.json"))
            checkpoint_docs = checkpoint_meta.get("docs_processed", 0)
            
            for event in sorted(new_preference_events, key=lambda x: x.get("at_docs", 0)):
                event_docs = event.get("at_docs", 0)
                if event_docs <= checkpoint_docs:
                    # Check if already applied
                    already_applied = any(
                        e.get("at_docs") == event_docs and e.get("type") == event.get("type")
                        for e in self.preference_history
                    )
                    if not already_applied:
                        self._handle_preference_event(event)
            
            # Re-run evaluation for this checkpoint
            print(f"\n🔄 Reprocessing checkpoint {checkpoint_id}...")
            metrics = self.run_checkpoint_evaluation(checkpoint_id, stream_dir)
            if metrics:
                updated_checkpoint_results.append(metrics)
        
        # Update stream results
        self.checkpoint_results = updated_checkpoint_results
        self._save_stream_results(stream_dir)
        
        print(f"\n✅ Reprocessing complete!")
        print(f"   Updated {len(updated_checkpoint_results)} checkpoints")
        
        return {
            "stream_dir": stream_dir,
            "checkpoints": updated_checkpoint_results,
            "metadata": self.stream_meta
        }
    
    def _restore_state_from_checkpoint(self, checkpoint_dir):
        """Restore stream state from a specific checkpoint"""
        # Load preference state
        pref_state_file = os.path.join(checkpoint_dir, "preference_state.json")
        if os.path.exists(pref_state_file):
            pref_state = self.utils.load_json(pref_state_file)
            self.active_preferences = pref_state.get("active_preferences", [])
            self.inactive_preferences = pref_state.get("inactive_preferences", [])
            self.preference_history = pref_state.get("preference_history", [])
            self._compute_preference_embeddings()
        
        # Load chunk metadata from kept.jsonl
        kept_file = os.path.join(checkpoint_dir, "kept.jsonl")
        if os.path.exists(kept_file):
            self.chunk_metadata = []
            self.preference_to_chunks = {}
            
            with open(kept_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        chunk_id = item.get("id", len(self.chunk_metadata))
                        metadata = {
                            "id": chunk_id,
                            "text": item["text"],
                            "instruction": item.get("instruction"),
                            "relevant_preferences": item.get("relevant_preferences", []),
                            "active": True,
                            "added_at_docs": item.get("added_at_docs", 0)
                        }
                        if item.get("processing_time") is not None:
                            metadata["processing_time"] = item["processing_time"]
                        
                        self.chunk_metadata.append(metadata)
                        
                        # Update preference_to_chunks mapping
                        for pref in metadata["relevant_preferences"]:
                            if pref not in self.preference_to_chunks:
                                self.preference_to_chunks[pref] = set()
                            self.preference_to_chunks[pref].add(chunk_id)
            
            self.next_chunk_id = max([m["id"] for m in self.chunk_metadata], default=0) + 1
            
            # Rebuild FAISS index from saved embeddings
            root_dir = self.utils.root_dir
            doc_mode_stream = f"{self.doc_mode}_stream"
            
            # Determine method suffix
            if self.utils.llm_model_name == "openai/gpt-oss-20b":
                llm_suffix = "_oss"
            elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
                llm_suffix = "_qwen"
            else:
                llm_suffix = ""
            
            # Build data_dir path for stream mode
            if self.utils.dataset_name == "PrefWiki":
                method_name = f"{self.method}_prefwiki{llm_suffix}"
            elif self.utils.dataset_name == "PrefELI5":
                method_name = f"{self.method}_prefeli5{llm_suffix}"
            elif self.utils.dataset_name == "PrefRQ":
                method_name = f"{self.method}_rq{llm_suffix}"
            elif self.utils.dataset_name == "PrefEval":
                method_name = f"{self.method}_prefeval{llm_suffix}"
            else:
                method_name = f"{self.method}_prefwiki{llm_suffix}"
            
            data_dir = os.path.join(root_dir, "indexing", doc_mode_stream, method_name, str(self.persona_index))
            model_name_clean = self.emb_model_name.replace('/', '_')
            embeddings_file = os.path.join(data_dir, f"embeddings_{model_name_clean}.npy")
            index_file = os.path.join(data_dir, f"index_{model_name_clean}.faiss")
            
            # Try to load FAISS index directly
            if os.path.exists(index_file):
                try:
                    self.faiss_index = faiss.read_index(index_file)
                    print(f"✅ Restored FAISS index from {index_file} ({self.faiss_index.ntotal} vectors)")
                    
                    # Load embeddings and update metadata
                    if os.path.exists(embeddings_file):
                        embeddings_array = np.load(embeddings_file)
                        # Match embeddings with metadata by chunk_id
                        sorted_metadata = sorted(self.chunk_metadata, key=lambda x: x["id"])
                        for i, meta in enumerate(sorted_metadata):
                            if i < len(embeddings_array):
                                meta["embedding"] = embeddings_array[i]
                except Exception as e:
                    print(f"⚠️ Failed to load FAISS index: {e}")
                    # Fallback: initialize empty index
                    if self.faiss_index is None:
                        self._init_faiss_index(self.embedding_dim)
            else:
                # No saved index, initialize empty
                if self.faiss_index is None:
                    self._init_faiss_index(self.embedding_dim)
                    print(f"⚠️ No saved FAISS index found, initialized empty index")


class StreamManager:
    """
    Manager class for running stream experiments
    """
    
    def __init__(self, utils):
        self.utils = utils
    
    def run_stream_experiment(self, persona_index, all_chunks, all_embeddings, 
                              method_dir, batch_size=2000, preference_events=None, skip_evaluation=False, stream_seed=None):
        """
        Run a complete stream experiment
        
        Args:
            persona_index: Target persona index
            all_chunks: All document chunks
            all_embeddings: All document embeddings
            method_dir: Output directory
            batch_size: Documents per batch
            preference_events: List of preference change events
            skip_evaluation: Skip evaluation during checkpoints (default: False)
            stream_seed: Random seed for preference events (for reproducibility)
        
        Returns:
            StreamSetup instance with results
        """
        stream = StreamSetup(self.utils, batch_size=batch_size, skip_evaluation=skip_evaluation, stream_seed=stream_seed)
        stream.initialize_stream(persona_index, all_chunks, all_embeddings)
        stream.run_stream(method_dir, preference_events)
        
        return stream
    
    def create_random_preference_events(self, num_add=2, num_remove=1, 
                                        total_docs=10000, batch_size=2000):
        """
        Create random preference events for testing
        
        Args:
            num_add: Number of preference add events
            num_remove: Number of preference remove events  
            total_docs: Total documents to process
            batch_size: Batch size for determining event timing
        
        Returns:
            list: Preference events
        """
        events = []
        
        # Generate random "add" events
        for _ in range(num_add):
            at_docs = random.randint(batch_size, total_docs - batch_size)
            events.append({
                "type": "add",
                "at_docs": at_docs,
                "preference": None  # Will get random from other persona
            })
        
        # Generate random "remove" events
        for _ in range(num_remove):
            at_docs = random.randint(batch_size * 2, total_docs - batch_size)
            events.append({
                "type": "remove",
                "at_docs": at_docs,
                "preference": None  # Will remove random active preference
            })
        
        return sorted(events, key=lambda x: x["at_docs"])
    
    def create_fixed_preference_events(self, batch_size=2000, total_docs=10000, 
                                       num_add=1, num_remove=1, seed=None):
        """
        Create random preference events avoiding checkpoint timings
        
        Preference events are randomly placed in the document stream,
        but avoid checkpoint evaluation timings to ensure stable evaluation.
        
        Args:
            batch_size: Batch size (checkpoint interval)
            total_docs: Total number of documents
            num_add: Number of preference add events
            num_remove: Number of preference remove events
            seed: Random seed for reproducibility
        
        Returns:
            list: Preference events sorted by at_docs
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Calculate checkpoint timings
        checkpoint_timings = set()
        checkpoint_interval = batch_size
        for i in range(1, (total_docs // checkpoint_interval) + 1):
            checkpoint_timings.add(i * checkpoint_interval)
        
        # Also avoid a window around checkpoints (±100 docs)
        checkpoint_windows = set()
        for cp_time in checkpoint_timings:
            for offset in range(-100, 101):
                checkpoint_windows.add(cp_time + offset)
        
        # Generate random event timings avoiding checkpoint windows
        events = []
        min_docs = max(100, checkpoint_interval)  # Start after first checkpoint
        max_docs = total_docs - checkpoint_interval  # End before last checkpoint
        
        # Generate "remove" events
        for _ in range(num_remove):
            attempts = 0
            while attempts < 100:  # Try up to 100 times
                at_docs = random.randint(min_docs, max_docs)
                if at_docs not in checkpoint_windows:
                    events.append({
                        "type": "remove",
                        "at_docs": at_docs,
                        "preference": None  # Will remove random active preference
                    })
                    break
                attempts += 1
            if attempts >= 100:
                # Fallback: use midpoint between checkpoints
                mid_point = (min_docs + max_docs) // 2
                events.append({
                    "type": "remove",
                    "at_docs": mid_point,
                    "preference": None
                })
        
        # Generate "add" events
        for _ in range(num_add):
            attempts = 0
            while attempts < 100:
                at_docs = random.randint(min_docs, max_docs)
                if at_docs not in checkpoint_windows:
                    events.append({
                        "type": "add",
                        "at_docs": at_docs,
                        "preference": None  # Will get random from other persona
                    })
                    break
                attempts += 1
            if attempts >= 100:
                mid_point = (min_docs + max_docs) // 2
                events.append({
                    "type": "add",
                    "at_docs": mid_point,
                    "preference": None
                })
        
        # Sort by at_docs
        events = sorted(events, key=lambda x: x["at_docs"])
        
        print(f"📅 Generated {len(events)} preference events (avoiding checkpoint timings)")
        for event in events:
            print(f"   - {event['type'].upper()} at {event['at_docs']} docs")
        
        return events

