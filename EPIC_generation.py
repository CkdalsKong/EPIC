import os
import json
import time
import faiss
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class EPICGeneration:
    def __init__(self, utils):
        self.utils = utils
        self.method = utils.method
        self.device = utils.device
        self.output_dir = utils.output_dir
        self.emb_model_name = utils.emb_model_name
        self.doc_mode = utils.doc_mode
        self.chunk_file = utils.chunk_file
        self.embedding_file = utils.embedding_file
        self.batch_size = getattr(utils, 'batch_size', 16)  # Default batch size if not set

    def process_query(self, query, preference_text, preferences, index, method_dir, generation_prompt, chunk_metadata=None):
        """
        Process a single query using unified FAISS index with metadata filtering
        
        Args:
            query: Query dict with "question" field
            preference_text: Current preference text
            preferences: List of all preferences
            index: Unified FAISS index
            method_dir: Method output directory
            generation_prompt: Prompt template for generation
            chunk_metadata: List of chunk metadata dicts with preference_ids
        """
        question = query["question"]
        
        start_retrieval = time.time()
        
        # Get query embedding
        query_emb = self.utils.embed_query_mp(question)
        
        # For standard method: use retrieve_top_k (mydata style)
        if self.method == "standard":
            filtered_chunks = [m["text"] for m in chunk_metadata] if chunk_metadata else []
            retrieved, retrieval_time = self.utils.retrieve_top_k(
                question,
                index,
                filtered_chunks
            )
            context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved)])
        
        # For insight/inst methods: find top-1 preference and filter by preference_ids
        elif self.method in ["EPIC_inst", "EPIC_inst_combined", "EPIC_insight", "EPIC_insight_combined"] and chunk_metadata is not None:
            # Find top-1 preference for this query
            preference_embs = []
            for pref in preferences:
                pref_emb = self.utils.embed_query_mp(pref)
                preference_embs.append(pref_emb.squeeze(0))
            
            preference_embs = np.vstack(preference_embs)
            sims = np.dot(preference_embs, query_emb.T).squeeze()
            top_pref_idx = np.argmax(sims)
            top_pref_text = preferences[top_pref_idx]
            
            # Augment query with preference
            query_emb = query_emb + self.utils.embed_query_mp(top_pref_text)
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            
            # Search with larger k to allow for filtering
            top_k = self.utils.top_k
            search_k = min(top_k * 5, index.ntotal)  # Fetch more to filter
            D, I = index.search(query_emb.astype(np.float32), search_k)
            
            # Filter by preference_ids and collect results
            retrieved = []
            retrieved_instructions = []
            for idx in I[0]:
                if idx < 0 or idx >= len(chunk_metadata):
                    continue
                meta = chunk_metadata[idx]
                # Check if this chunk is relevant to top preference
                if top_pref_text in meta.get("preference_ids", []):
                    retrieved.append(meta["text"])
                    retrieved_instructions.append(meta.get("instruction", meta.get("insight", "")))
                    if len(retrieved) >= top_k:
                        break
            
            retrieval_time = time.time() - start_retrieval
            
            # Build context with instructions/insights
            context_parts = []
            for i, (doc, inst) in enumerate(zip(retrieved, retrieved_instructions)):
                if inst:
                    context_parts.append(f"Document {i+1}:\nInterpretation Guidance: {inst}\nContent: {doc}")
                else:
                    context_parts.append(f"Document {i+1}: {doc}")
            context = "\n\n".join(context_parts)
        elif self.method == "cosine":
            # Cosine method: use retrieve_top_k (mydata style)
            filtered_chunks = [m["text"] for m in chunk_metadata] if chunk_metadata else []
            retrieved, retrieval_time = self.utils.retrieve_top_k(
                question,
                index,
                filtered_chunks
            )
            context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved)])
        else:
            # Other methods: use retrieve_top_k_wq_cosine
            filtered_chunks = [m["text"] for m in chunk_metadata] if chunk_metadata else []
            retrieved, retrieval_time = self.utils.retrieve_top_k_wq_cosine(
                question,
                preferences,
                index,
                filtered_chunks
            )
            context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved)])
        
        filled_prompt = generation_prompt.replace("{context}", context).replace("{question}", question)
        
        try:
            if self.utils.llm_model_name == "openai/gpt-oss-20b":
                max_tokens = 8192
            else:
                max_tokens = 2048
            generated_text = self.utils.generate_message_vllm(
                messages=[{"role": "user", "content": filled_prompt}],
                system_prompt="You are a helpful assistant for generating responses.",
                max_tokens=max_tokens
            )
            
            if generated_text is None:
                print(f"Warning: LLM returned None response for generation - returning None")
                return None
            
            # Return result if successfully generated
            return {
                "preference": preference_text,
                "question": question,
                "response_to_q": generated_text,
                "retrieved_docs": retrieved,
                "retrieval_time": retrieval_time
            }
            
        except Exception as e:
            print(f"Failed to generate response: {e} - returning None")
            return None
    
    def run_generation_with_cache(self, persona_index, method_dir, cached_resources):
        print(f"\n=== Starting generation for persona {persona_index} ===")

        # For standard method: use common directory (mydata style)
        if self.method == "standard":
            data_dir = self.utils.data_dir
        else:
            if self.utils.dataset_name == "PrefWiki":
                data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
            elif self.utils.dataset_name == "PrefELI5":
                data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
            elif self.utils.dataset_name == "PrefRQ":
                data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
            elif self.utils.dataset_name == "PrefEval":
                data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
            
        # Determine index file path based on index type
        model_name_clean = self.emb_model_name.replace("/", "_")

        # Load data
        persona = self.utils.load_persona_data(persona_index)
        print(f"Loaded persona data for index {persona_index}")
        
        preference_list = [block["preference"] for block in persona["preference_blocks"]]

        # Load unified FAISS index and metadata (same approach for all methods)
        # For standard method: use data_dir directly (mydata style)
        if self.method == "standard":
            index_file = os.path.join(data_dir, f"index_{model_name_clean}.faiss")
        else:
            index_file = os.path.join(method_dir, f"index_{model_name_clean}.faiss")
            if not os.path.exists(index_file):
                # Fallback to data_dir if not in method_dir
                index_file = os.path.join(data_dir, f"index_{model_name_clean}.faiss")
        
        print(f"Loading FAISS index from: {index_file}")
        index = faiss.read_index(index_file)
        print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
        
        # Load chunk metadata with preference_ids
        # For standard method: use method_dir directly (mydata style)
        if self.method == "standard":
            kept_file = os.path.join(method_dir, "kept.jsonl")
        else:
            kept_file = os.path.join(method_dir, "kept.jsonl")
        
        chunk_metadata = []
        
        if os.path.exists(kept_file):
            with open(kept_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    chunk_metadata.append({
                        "text": item["text"],
                        # Handle different field names from different methods
                        "preference_ids": item.get("relevant_preferences", item.get("preference_ids", item.get("relevant_preference", []))),
                        "instruction": item.get("instruction", ""),
                        "insight": item.get("insight", ""),
                        "reason": item.get("reason", "")
                    })
            
            print(f"✅ Loaded {len(chunk_metadata)} chunks with metadata")
        else:
            print(f"⚠️ No kept.jsonl found at {kept_file}")
        
        # Count chunks per preference for insight/inst methods
        if self.method in ["EPIC_inst", "EPIC_inst_combined", "EPIC_insight", "EPIC_insight_combined"]:
            pref_counts = {}
            for meta in chunk_metadata:
                for pref in meta.get("preference_ids", []):
                    pref_counts[pref[:50] + "..."] = pref_counts.get(pref[:50] + "...", 0) + 1
            print(f"Chunks per preference: {pref_counts}")
        
        generation_prompt = self.utils.load_prompt_template(self.utils.generation_prompt)

        all_results = []
        retrieval_times = []

        # Use cached models (skip model loading)
        print("✅ Using cached models")

        # Process each preference block
        for block in persona["preference_blocks"]:
            preference_text = block["preference"]
            preferences = [block["preference"] for block in persona["preference_blocks"]]
            queries = block["queries"]
            
            # Process each query with ThreadPoolExecutor (CUDA compatible)
            with ThreadPoolExecutor(max_workers=1) as executor:  # Set workers to number of GPUs
                futures = []
                for query in queries:
                    # Use unified FAISS with metadata for all methods
                    future = executor.submit(
                        self.process_query,
                        query,
                        preference_text,
                        preferences,
                        index,
                        method_dir,
                        generation_prompt,
                        chunk_metadata
                    )
                    futures.append(future)
                
                # Collect results
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing queries for preference: {preference_text[:50]}..."):
                    result = future.result()
                    if result:
                        all_results.append(result)
                        retrieval_times.append(result["retrieval_time"])
                    
                    # Enhanced memory cleanup
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        output_file = os.path.join(method_dir, f"gen_{self.method}_flat_{persona_index}.json")
        
        self.utils.save_json(output_file, all_results)
        print(f"✅ Generation results saved to {output_file}")

        # Calculate retrieval time statistics
        avg_time = np.mean(retrieval_times)
        max_time = np.max(retrieval_times)
        min_time = np.min(retrieval_times)

        if self.utils.llm_model_name == "openai/gpt-oss-20b":
            llm_name = "_oss"
        elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            llm_name = "_qwen"
        else:
            llm_name = ""
        # Write report
        fieldnames = ["method", "persona_index", "avg_retrieval_time(s)", "max_retrieval_time(s)", "min_retrieval_time(s)"]
        row = {
            "method": f"{self.method}{llm_name}",
            "persona_index": f"{persona_index}",
            "avg_retrieval_time(s)": f"{avg_time:.4f}",
            "max_retrieval_time(s)": f"{max_time:.4f}",
            "min_retrieval_time(s)": f"{min_time:.4f}"
        }
        
        self.utils.save_csv(os.path.join(self.output_dir, self.utils.generation_report_file), fieldnames, row)
        
        print(f"\n=== Completed generation for persona {persona_index} ===")
        print(f"Average retrieval time: {avg_time:.4f} seconds")
        print(f"Max retrieval time: {max_time:.4f} seconds")
        print(f"Min retrieval time: {min_time:.4f} seconds")
        return method_dir