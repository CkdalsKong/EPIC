import os
import json
import time
import faiss
import torch
import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    
    def __init__(self, utils, batch_size=2000):
        self.utils = utils
        self.method = utils.method
        self.device = utils.device
        self.output_dir = utils.output_dir
        self.emb_model_name = utils.emb_model_name
        self.doc_mode = utils.doc_mode
        self.batch_size = batch_size  # Documents per batch (default: 2000)
        
        # Stream state - using metadata-based management
        self.chunk_metadata = []  # [{id, text, insight, relevant_preferences, active}]
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
        
        # Load persona data
        self.persona = self.utils.load_persona_data(persona_index)
        self.active_preferences = [block["preference"] for block in self.persona["preference_blocks"]]
        
        # Initialize preference embeddings
        self._compute_preference_embeddings()
        
        # Reset stream state
        self.chunk_metadata = []
        self.next_chunk_id = 0
        self.checkpoint_results = []
        self.preference_to_chunks = {pref: set() for pref in self.active_preferences}
        
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
        LLM filtering to decide Keep/Discard for a chunk
        
        Args:
            chunk: Document chunk text
            relevant_preferences: List of relevant preference texts (from cosine filtering)
        
        Returns:
            tuple: (keep: bool, filtered_preferences: list, reason: str)
        """
        try:
            # Format preferences for prompt
            preference_text = "\n".join([f"{i+1}. '{p}'" for i, p in enumerate(relevant_preferences)])
            
            # Load filtering prompts based on method
            if self.method == "EPIC_insight_combined":
                system_prompt = self.utils.load_prompt_template(self.utils.filtering_insight_system)
                user_prompt = self.utils.load_prompt_template(self.utils.filtering_insight_user)
            elif self.method == "EPIC_insight":
                system_prompt = self.utils.load_prompt_template(self.utils.filtering_inst_system)
                user_prompt = self.utils.load_prompt_template(self.utils.filtering_inst_user)
            else:
                # Default filtering prompt
                system_prompt = self.utils.load_prompt_template(self.utils.filtering_inst_system)
                user_prompt = self.utils.load_prompt_template(self.utils.filtering_inst_user)
            
            filled_prompt = self.utils.format_prompt(user_prompt, preference_text, chunk)
            
            response = self.utils.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt=system_prompt,
                max_tokens=512
            )
            
            if response is None:
                return False, [], "LLM returned None"
            
            # Parse decision
            decision, reason, preferences = self.utils.parse_decision_and_reason_preferences(response)
            
            # Map numbered preferences to actual text
            if preferences:
                mapped_prefs = []
                for pref in preferences:
                    mapped = self.utils.map_preference_numbers_to_text(pref, relevant_preferences)
                    if mapped:
                        mapped_prefs.append(mapped)
                preferences = mapped_prefs
            
            keep = decision == "Keep"
            return keep, preferences if preferences else [], reason
            
        except Exception as e:
            print(f"LLM filtering failed: {e}")
            return False, [], f"Error: {str(e)}"
    
    def _generate_insight_for_chunk(self, chunk, relevant_preferences, reason=""):
        """
        Generate insight for a single chunk using LLM
        
        Args:
            chunk: Document chunk text
            relevant_preferences: List of relevant preference texts
            reason: Reason from filtering step
        
        Returns:
            str: Generated insight
        """
        try:
            preference_text = "\n".join([f"- {p}" for p in relevant_preferences])
            
            system_prompt = self.utils.load_prompt_template(self.utils.insight_system)
            user_prompt = self.utils.load_prompt_template(self.utils.insight_user)
            
            filled_prompt = user_prompt.format(
                preference=preference_text, 
                chunk=chunk, 
                reason=reason if reason else "Matched by LLM filtering"
            )
            
            response = self.utils.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt=system_prompt,
                max_tokens=256
            )
            
            if response:
                insight = self.utils.parse_insight(response)
                return insight if insight else f"Relevant to: {preference_text[:100]}..."
            else:
                return f"Relevant to: {preference_text[:100]}..."
                
        except Exception as e:
            print(f"Insight generation failed: {e}")
            return f"Relevant to: {', '.join(relevant_preferences)[:100]}..."
    
    def _llm_filter_and_insight_combined(self, chunk, relevant_preferences):
        """
        Combined LLM call for filtering and insight generation (for EPIC_insight_combined)
        
        Args:
            chunk: Document chunk text
            relevant_preferences: List of relevant preference texts
        
        Returns:
            tuple: (keep: bool, filtered_preferences: list, insight: str)
        """
        try:
            preference_text = "\n".join([f"{i+1}. '{p}'" for i, p in enumerate(relevant_preferences)])
            
            system_prompt = self.utils.load_prompt_template(self.utils.filtering_insight_system)
            user_prompt = self.utils.load_prompt_template(self.utils.filtering_insight_user)
            
            filled_prompt = self.utils.format_prompt(user_prompt, preference_text, chunk)
            
            response = self.utils.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt=system_prompt,
                max_tokens=512
            )
            
            if response is None:
                return False, [], ""
            
            # Parse decision, reason, preferences, and insight
            decision, reason, preferences, insight = self.utils.parse_decision_reason_insight(response)
            
            # Map numbered preferences to actual text
            if preferences:
                mapped_prefs = []
                for pref in preferences:
                    mapped = self.utils.map_preference_numbers_to_text(pref, relevant_preferences)
                    if mapped:
                        mapped_prefs.append(mapped)
                preferences = mapped_prefs
            
            keep = decision == "Keep"
            return keep, preferences if preferences else [], insight if insight else ""
            
        except Exception as e:
            print(f"Combined LLM filter+insight failed: {e}")
            return False, [], ""
    
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
            
            # Reactivate chunks that belong to this preference
            if preference_text in self.preference_to_chunks:
                chunk_ids = self.preference_to_chunks[preference_text]
                for chunk_id in chunk_ids:
                    if chunk_id < len(self.chunk_metadata):
                        meta = self.chunk_metadata[chunk_id]
                        if not meta["active"]:
                            meta["active"] = True
                            meta["reactivated_at_docs"] = self.stream_meta["total_docs_processed"]
                            reactivated_count += 1
        
        self.active_preferences.append(preference_text)
        self._compute_preference_embeddings()
        
        # Initialize preference_to_chunks if new preference
        if preference_text not in self.preference_to_chunks:
            self.preference_to_chunks[preference_text] = set()
        
        # Update active chunks count
        active_count = sum(1 for m in self.chunk_metadata if m["active"])
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
        self._compute_preference_embeddings()
        
        # Deactivate chunks that ONLY belong to this preference
        deactivated_count = 0
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
        active_count = sum(1 for m in self.chunk_metadata if m["active"])
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
    
    def process_batch(self, start_idx, end_idx):
        """
        Process a batch of documents with incremental FAISS updates
        
        Args:
            start_idx: Start index in all_chunks
            end_idx: End index in all_chunks
        
        Returns:
            dict: Batch processing results
        """
        batch_start_time = time.time()
        
        # Get batch data
        batch_chunks = self.all_chunks[start_idx:end_idx]
        batch_embeddings = self.all_embeddings[start_idx:end_idx]
        
        # Normalize embeddings
        batch_embeddings_norm = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        
        # Filter based on preference similarity (cosine filtering)
        kept_chunks = []
        kept_embeddings = []
        kept_insights = []
        kept_preferences = []  # List of relevant preferences for each kept chunk
        filtered_count = 0
        
        if self.preference_embeddings is not None and len(self.active_preferences) > 0:
            # Compute similarity with preferences
            sims = np.dot(self.preference_embeddings, batch_embeddings_norm.T)
            above_threshold = sims > self.utils.threshold
            mask = np.any(above_threshold, axis=0)
            
            for j, (chunk, emb, is_kept) in enumerate(zip(batch_chunks, batch_embeddings_norm, mask)):
                if is_kept:
                    # Track which preferences this document relates to
                    relevant_pref_indices = np.where(above_threshold[:, j])[0]
                    relevant_prefs = [self.active_preferences[idx] for idx in relevant_pref_indices]
                    
                    # For insight methods: generate insight using LLM
                    insight = None
                    if self.method in ["EPIC_insight", "EPIC_insight_combined"]:
                        insight = self._generate_insight_for_chunk(chunk, relevant_prefs)
                        kept_insights.append(insight)
                    
                    kept_chunks.append(chunk)
                    kept_embeddings.append(emb)
                    kept_preferences.append(relevant_prefs)
                else:
                    filtered_count += 1
        else:
            kept_chunks = batch_chunks
            kept_embeddings = list(batch_embeddings_norm)
            kept_preferences = [self.active_preferences[:] for _ in batch_chunks]
        
        # Incrementally add to FAISS index and metadata
        if kept_chunks:
            self._add_chunks_to_index(kept_chunks, kept_embeddings, kept_preferences, kept_insights)
        
        batch_time = time.time() - batch_start_time
        
        # Update metadata
        self.stream_meta["total_docs_processed"] += len(batch_chunks)
        active_count = sum(1 for m in self.chunk_metadata if m["active"])
        self.stream_meta["active_chunks"] = active_count
        
        result = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "batch_size": len(batch_chunks),
            "kept_count": len(kept_chunks),
            "filtered_count": filtered_count,
            "processing_time": batch_time,
            "total_indexed": len(self.chunk_metadata),
            "total_active": active_count
        }
        
        print(f"📦 Batch [{start_idx}:{end_idx}] - Kept: {len(kept_chunks)}, Filtered: {filtered_count}, Total indexed: {len(self.chunk_metadata)}, Active: {active_count}")
        
        return result
    
    def _add_chunks_to_index(self, chunks, embeddings, preferences_list, insights=None):
        """
        Incrementally add chunks to FAISS index with metadata
        
        Args:
            chunks: List of chunk texts
            embeddings: List of embedding vectors
            preferences_list: List of relevant preferences for each chunk
            insights: Optional list of insights for each chunk
        """
        if not chunks:
            return
        
        # Prepare IDs and embeddings for FAISS
        start_id = self.next_chunk_id
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
        emb_array = np.array(embeddings).astype(np.float32)
        
        # Add to FAISS index
        self.faiss_index.add_with_ids(emb_array, ids)
        
        # Add metadata for each chunk
        for i, (chunk, prefs) in enumerate(zip(chunks, preferences_list)):
            chunk_id = start_id + i
            insight = insights[i] if insights and i < len(insights) else None
            
            metadata = {
                "id": chunk_id,
                "text": chunk,  # "text" to match indexing
                "insight": insight,
                "relevant_preferences": prefs,  # "relevant_preferences" to match indexing
                "active": True,
                "added_at_docs": self.stream_meta["total_docs_processed"]
            }
            self.chunk_metadata.append(metadata)
            
            # Update preference_to_chunks mapping
            for pref in prefs:
                if pref not in self.preference_to_chunks:
                    self.preference_to_chunks[pref] = set()
                self.preference_to_chunks[pref].add(chunk_id)
        
        self.next_chunk_id += len(chunks)
        self.stream_meta["total_chunks_indexed"] = len(self.chunk_metadata)
    
    def get_index(self):
        """Get the current FAISS index (no rebuild needed with incremental updates)"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("⚠️ No embeddings in index")
            return None
        
        active_count = sum(1 for m in self.chunk_metadata if m["active"])
        print(f"✅ Using FAISS index with {self.faiss_index.ntotal} vectors ({active_count} active)")
        return self.faiss_index
    
    def search_active_chunks(self, query_emb, top_k=5):
        """
        Search FAISS index and return only active chunks
        
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
                if meta["active"]:
                    results.append(meta["text"])
                    if len(results) >= top_k:
                        break
        
        return results
    
    def get_active_chunks(self):
        """Get list of all active chunks"""
        return [m["text"] for m in self.chunk_metadata if m["active"]]
    
    def get_active_chunks_with_insights(self):
        """Get list of all active chunks with their insights"""
        return [(m["text"], m.get("insight", "")) for m in self.chunk_metadata if m["active"]]
    
    # Alias for backward compatibility
    def build_index(self):
        """Alias for get_index() - maintained for backward compatibility"""
        return self.get_index()
    
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
        index = self.get_index()
        if index is None:
            return None
        
        # Save current chunks for generation
        checkpoint_dir = os.path.join(method_dir, f"checkpoint_{checkpoint_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save active chunks (with insights if available)
        kept_file = os.path.join(checkpoint_dir, "kept.jsonl")
        active_chunks = []
        with open(kept_file, 'w', encoding='utf-8') as f:
            for meta in self.chunk_metadata:
                if meta["active"]:
                    item = {"text": meta["text"], "id": meta["id"]}
                    if meta.get("insight"):
                        item["insight"] = meta["insight"]
                    item["relevant_preferences"] = meta["relevant_preferences"]
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    active_chunks.append(meta["text"])
        
        # Run generation for a subset of queries
        generation_results = self._run_generation(index, checkpoint_dir, active_chunks)
        
        # Run evaluation on generated results
        evaluation_stats = self._run_evaluation(generation_results, checkpoint_dir)
        
        # Calculate metrics
        total = len(generation_results) if generation_results else 1
        active_count = len(active_chunks)
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
            "preference_following_accuracy": round((evaluation_stats.get("preference_adherence_accuracy", 0) / total) * 100, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        self.checkpoint_results.append(metrics)
        self.stream_meta["checkpoints"].append(metrics)
        
        # Save checkpoint results
        self._save_checkpoint_results(checkpoint_dir, metrics)
        
        print(f"✅ Checkpoint #{checkpoint_id} evaluation complete")
        print(f"   Total indexed: {len(self.chunk_metadata)}, Active: {active_count}")
        print(f"   Accuracy: {metrics['preference_following_accuracy']}%")
        
        return metrics
    
    def _run_generation(self, index, checkpoint_dir, active_chunks=None):
        """Run generation for current checkpoint"""
        generation_prompt = self.utils.load_prompt_template(self.utils.generation_prompt)
        all_results = []
        
        # Get active chunks if not provided
        if active_chunks is None:
            active_chunks = self.get_active_chunks()
        
        # Process each active preference block
        for block in self.persona["preference_blocks"]:
            preference_text = block["preference"]
            
            # Skip if preference is inactive
            if preference_text not in self.active_preferences:
                continue
            
            queries = block["queries"][:3]  # Limit queries per checkpoint for speed
            
            for query in queries:
                question = query["question"]
                
                try:
                    start_retrieval = time.time()
                    
                    # Use active-aware retrieval
                    retrieved = self._retrieve_active_chunks(question, self.utils.top_k)
                    retrieval_time = time.time() - start_retrieval
                    
                    if not retrieved:
                        # Fallback to standard retrieval if no results
                        retrieved, retrieval_time = self.utils.retrieve_top_k_wq_cosine(
                            question,
                            self.active_preferences,
                            index,
                            active_chunks
                        )
                    
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
        Retrieve top-k active chunks for a question
        
        Args:
            question: Query question
            top_k: Number of results
        
        Returns:
            list: Retrieved chunk texts
        """
        # Compute query embedding
        query_emb = self.utils.embed_query_mp(question)
        
        # Add preference weighting
        if self.preference_embeddings is not None and len(self.active_preferences) > 0:
            pref_sims = np.dot(self.preference_embeddings, query_emb.T).squeeze()
            max_pref_idx = np.argmax(pref_sims)
            pref_emb = self.utils.embed_query_mp(self.active_preferences[max_pref_idx])
            query_emb = query_emb + pref_sims[max_pref_idx] * pref_emb
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        
        # Search and filter by active status
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
        active_count = sum(1 for m in self.chunk_metadata if m["active"])
        inactive_count = len(self.chunk_metadata) - active_count
        self.utils.save_json(chunk_summary_file, {
            "total_indexed": len(self.chunk_metadata),
            "active": active_count,
            "inactive": inactive_count
        })
    
    def run_stream(self, method_dir, preference_events=None, checkpoint_interval=None):
        """
        Run the full stream processing pipeline (document-by-document)
        
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
        
        stream_dir = os.path.join(method_dir, f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(stream_dir, exist_ok=True)
        
        total_docs = len(self.all_chunks)
        checkpoint_interval = checkpoint_interval or self.batch_size
        num_checkpoints = (total_docs + checkpoint_interval - 1) // checkpoint_interval
        
        print(f"Total documents: {total_docs}")
        print(f"Processing: Document-by-document")
        print(f"Checkpoint interval: Every {checkpoint_interval} documents")
        print(f"Expected checkpoints: {num_checkpoints}")
        
        # Sort preference events by docs threshold
        if preference_events:
            preference_events = sorted(preference_events, key=lambda x: x.get("at_docs", 0))
        else:
            preference_events = []
        
        event_idx = 0
        checkpoint_id = 0
        docs_since_checkpoint = 0
        
        # Process documents one by one
        for doc_idx in tqdm(range(total_docs), desc="Processing documents"):
            # Check for preference events
            while event_idx < len(preference_events):
                event = preference_events[event_idx]
                if event.get("at_docs", 0) <= self.stream_meta["total_docs_processed"]:
                    self._handle_preference_event(event)
                    event_idx += 1
                else:
                    break
            
            # Process single document
            self._process_single_document(doc_idx)
            docs_since_checkpoint += 1
            
            # Run checkpoint at intervals
            if docs_since_checkpoint >= checkpoint_interval:
                checkpoint_id += 1
                print(f"\n--- Checkpoint {checkpoint_id} at {self.stream_meta['total_docs_processed']} docs ---")
                self.run_checkpoint_evaluation(checkpoint_id, stream_dir)
                docs_since_checkpoint = 0
        
        # Final checkpoint if there are remaining docs
        if docs_since_checkpoint > 0:
            checkpoint_id += 1
            print(f"\n--- Final Checkpoint {checkpoint_id} at {self.stream_meta['total_docs_processed']} docs ---")
            self.run_checkpoint_evaluation(checkpoint_id, stream_dir)
        
        # Save final stream results
        self._save_stream_results(stream_dir)
        
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
            2. LLM filtering - ask LLM to decide Keep/Discard
            3. If Keep -> Insight generation (for insight methods)
            4. Add to FAISS index
        """
        chunk = self.all_chunks[doc_idx]
        emb = self.all_embeddings[doc_idx]
        
        # Normalize embedding
        emb_norm = emb / np.linalg.norm(emb)
        
        # Update processed count first
        self.stream_meta["total_docs_processed"] += 1
        
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
        
        # If cosine filtering failed, skip this document
        if not cosine_passed:
            return
        
        # ============================================
        # Step 2: LLM Filtering (Keep/Discard)
        # ============================================
        # For methods that need LLM filtering
        if self.method in ["EPIC_insight", "EPIC_insight_combined", "EPIC_inst", "EPIC_inst_combined"]:
            if self.method == "EPIC_insight_combined":
                # Combined: LLM filtering + insight in one call
                llm_keep, llm_prefs, insight = self._llm_filter_and_insight_combined(chunk, cosine_relevant_prefs)
                
                if not llm_keep:
                    return  # Discard
                
                # Use LLM-filtered preferences, or fall back to cosine prefs
                final_prefs = llm_prefs if llm_prefs else cosine_relevant_prefs
                
            elif self.method == "EPIC_insight":
                # Separate: LLM filtering first, then insight generation
                llm_keep, llm_prefs, reason = self._llm_filter_chunk(chunk, cosine_relevant_prefs)
                
                if not llm_keep:
                    return  # Discard
                
                final_prefs = llm_prefs if llm_prefs else cosine_relevant_prefs
                
                # Generate insight separately
                insight = self._generate_insight_for_chunk(chunk, final_prefs, reason)
            else:
                # EPIC_inst, EPIC_inst_combined - just filtering, no insight
                llm_keep, llm_prefs, reason = self._llm_filter_chunk(chunk, cosine_relevant_prefs)
                
                if not llm_keep:
                    return  # Discard
                
                final_prefs = llm_prefs if llm_prefs else cosine_relevant_prefs
                insight = None
        else:
            # For cosine-only method, no LLM filtering
            final_prefs = cosine_relevant_prefs
            insight = None
        
        # ============================================
        # Step 3: Add to FAISS Index
        # ============================================
        # For insight methods: use insight embedding for FAISS
        if self.method in ["EPIC_insight", "EPIC_insight_combined"] and insight:
            insight_emb = self.utils.embed_query_mp(insight)
            insight_emb = insight_emb / np.linalg.norm(insight_emb, axis=1, keepdims=True)
            self._add_single_chunk_to_index(chunk, insight_emb.squeeze(0), final_prefs, insight)
        else:
            # For other methods: use chunk embedding
            self._add_single_chunk_to_index(chunk, emb_norm, final_prefs, insight)
    
    def _add_single_chunk_to_index(self, chunk, embedding, preferences, insight=None):
        """
        Add a single chunk to the FAISS index
        
        Args:
            chunk: Chunk text
            embedding: Normalized embedding vector
            preferences: List of relevant preferences
            insight: Optional insight text
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
            "insight": insight,
            "relevant_preferences": preferences,  # "relevant_preferences" to match indexing
            "active": True,
            "added_at_docs": self.stream_meta["total_docs_processed"]
        }
        self.chunk_metadata.append(metadata)
        
        # Update preference_to_chunks mapping
        for pref in preferences:
            if pref not in self.preference_to_chunks:
                self.preference_to_chunks[pref] = set()
            self.preference_to_chunks[pref].add(chunk_id)
        
        self.next_chunk_id += 1
        self.stream_meta["total_chunks_indexed"] = len(self.chunk_metadata)
        self.stream_meta["active_chunks"] = sum(1 for m in self.chunk_metadata if m["active"])
    
    def _handle_preference_event(self, event):
        """Handle a preference change event"""
        event_type = event.get("type")
        preference = event.get("preference")
        
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


class StreamManager:
    """
    Manager class for running stream experiments
    """
    
    def __init__(self, utils):
        self.utils = utils
    
    def run_stream_experiment(self, persona_index, all_chunks, all_embeddings, 
                              method_dir, batch_size=2000, preference_events=None):
        """
        Run a complete stream experiment
        
        Args:
            persona_index: Target persona index
            all_chunks: All document chunks
            all_embeddings: All document embeddings
            method_dir: Output directory
            batch_size: Documents per batch
            preference_events: List of preference change events
        
        Returns:
            StreamSetup instance with results
        """
        stream = StreamSetup(self.utils, batch_size=batch_size)
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
    
    def create_fixed_preference_events(self, batch_size=2000):
        """
        Create fixed preference events at specific checkpoints
        
        - 2nd checkpoint (after 2 batches): Remove one preference
        - 4th checkpoint (after 4 batches): Add one preference
        
        Args:
            batch_size: Batch size for determining event timing
        
        Returns:
            list: Preference events
        """
        events = [
            {
                "type": "remove",
                "at_docs": batch_size * 2,  # After 2nd batch (e.g., 4000 docs)
                "preference": None  # Will remove random active preference
            },
            {
                "type": "add",
                "at_docs": batch_size * 4,  # After 4th batch (e.g., 8000 docs)
                "preference": None  # Will get random from other persona
            }
        ]
        
        return events

