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

    def process_query(self, query, preference_text, preferences, filtered_chunks, index, method_dir, generation_prompt, filtered_instructions=None, pref_indices=None, pref_chunks_map=None, pref_instructions_map=None):
        question = query["question"]

        # For EPIC_inst and EPIC_inst_combined: use preference-specific FAISS
        if self.method in ["EPIC_inst", "EPIC_inst_combined"] and pref_indices is not None:
            # Find top-1 preference for this query
            query_emb = self.utils.embed_query_mp(question)
            preference_embs = []
            for pref in preferences:
                pref_emb = self.utils.embed_query_mp(pref)
                preference_embs.append(pref_emb.squeeze(0))
            
            preference_embs = np.vstack(preference_embs)
            sims = np.dot(preference_embs, query_emb.T).squeeze()
            top_pref_idx = np.argmax(sims)
            
            # Use the preference-specific index
            pref_index = pref_indices[top_pref_idx]
            pref_chunks = pref_chunks_map[top_pref_idx]
            pref_instructions = pref_instructions_map[top_pref_idx]
            
            # Retrieve from preference-specific FAISS
            start_retrieval = time.time()
            query_emb = self.utils.embed_query_mp(question)
            query_emb = query_emb + self.utils.embed_query_mp(preferences[top_pref_idx])
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            
            top_k = self.utils.top_k
            D, I = pref_index.search(query_emb, top_k)
            retrieval_time = time.time() - start_retrieval
            
            retrieved = [pref_chunks[i] for i in I[0]]
            retrieved_instructions = [pref_instructions[i] for i in I[0]]
            
            # Augment with instructions
            context_parts = []
            for i, (doc, inst) in enumerate(zip(retrieved, retrieved_instructions)):
                context_parts.append(f"Document {i+1}:\nInterpretation Guidance: {inst}\nContent: {doc}")
            context = "\n\n".join(context_parts)
        else:
            # Original method
            retrieved, retrieval_time = self.utils.retrieve_top_k_wq_cosine(
                question,
                preferences,
                index,
                filtered_chunks
            )
            
            # For EPIC_inst and EPIC_inst_combined with single index (legacy)
            if self.method in ["EPIC_inst", "EPIC_inst_combined"] and filtered_instructions is not None:
                # Find corresponding instructions for retrieved chunks
                context_parts = []
                for i, doc in enumerate(retrieved):
                    try:
                        doc_idx = filtered_chunks.index(doc)
                        instruction = filtered_instructions[doc_idx]
                        context_parts.append(f"Document {i+1}:\nInterpretation Guidance: {instruction}\nContent: {doc}")
                    except (ValueError, IndexError):
                        # Fallback if instruction not found
                        context_parts.append(f"Document {i+1}: {doc}")
                context = "\n\n".join(context_parts)
            else:
                # Original method: just concatenate documents
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

        if self.utils.dataset_name == "PrefWiki":
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        elif self.utils.dataset_name == "PrefELI5":
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        elif self.utils.dataset_name == "PrefRQ":
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
            
        # Determine index file path based on index type
        model_name_clean = self.emb_model_name.replace("/", "_")

        # Load data
        persona = self.utils.load_persona_data(persona_index)
        print(f"Loaded persona data for index {persona_index}")
        
        preference_list = [block["preference"] for block in persona["preference_blocks"]]

        # For EPIC_inst and EPIC_inst_combined: load preference-specific FAISS indices
        if self.method in ["EPIC_inst", "EPIC_inst_combined"]:
            # Load preference mapping
            pref_mapping_file = os.path.join(method_dir, "preference_mapping.json")
            if os.path.exists(pref_mapping_file):
                print("Loading preference-specific FAISS indices...")
                with open(pref_mapping_file, 'r', encoding='utf-8') as f:
                    pref_mapping = json.load(f)
                
                pref_to_idx = pref_mapping["preference_to_idx"]
                
                # Load each preference's FAISS index and chunks
                pref_indices = {}
                pref_chunks_map = {}
                pref_instructions_map = {}
                
                for pref_text, pref_idx in pref_to_idx.items():
                    pref_index_file = os.path.join(data_dir, f"index_pref{pref_idx}_{model_name_clean}.faiss")
                    pref_kept_file = os.path.join(method_dir, f"kept_pref{pref_idx}.jsonl")
                    
                    if os.path.exists(pref_index_file) and os.path.exists(pref_kept_file):
                        pref_index = faiss.read_index(pref_index_file)
                        pref_indices[pref_idx] = pref_index
                        
                        # Load chunks and instructions for this preference
                        pref_chunks = []
                        pref_instructions = []
                        with open(pref_kept_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                item = json.loads(line)
                                pref_chunks.append(item["text"])
                                pref_instructions.append(item.get("instruction", ""))
                        
                        pref_chunks_map[pref_idx] = pref_chunks
                        pref_instructions_map[pref_idx] = pref_instructions
                        print(f"  Loaded preference {pref_idx}: {len(pref_chunks)} chunks")
                    else:
                        print(f"  Warning: Missing files for preference {pref_idx}")
                
                print(f"✅ Loaded {len(pref_indices)} preference-specific FAISS indices")
                
                # Set variables for preference-based retrieval
                index = None
                filtered_chunks = None
                filtered_instructions = None
            else:
                print("⚠️ Preference mapping not found, falling back to single index")
                # Fallback to single index
                index_file = os.path.join(data_dir, f"index_{model_name_clean}.faiss")
                index = faiss.read_index(index_file)
                
                filtered_chunks_file = os.path.join(method_dir, "kept.jsonl")
                filtered_chunks = []
                filtered_instructions = []
                with open(filtered_chunks_file, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        filtered_chunks.append(item["text"])
                        filtered_instructions.append(item.get("instruction", ""))
                
                pref_indices = None
                pref_chunks_map = None
                pref_instructions_map = None
        else:
            # Original method: single FAISS index
            index_file = os.path.join(data_dir, f"index_{model_name_clean}.faiss")
            index = faiss.read_index(index_file)

            filtered_chunks_file = os.path.join(method_dir, "kept.jsonl")
            filtered_chunks = []
            filtered_instructions = []
            
            with open(filtered_chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    filtered_chunks.append(item["text"])
                    # For EPIC_inst and EPIC_inst_combined, also load instructions
                    if self.method in ["EPIC_inst", "EPIC_inst_combined"]:
                        filtered_instructions.append(item.get("instruction", ""))
            
            print(f"Loaded {len(filtered_chunks)} filtered chunks")
            if self.method in ["EPIC_inst", "EPIC_inst_combined"]:
                print(f"Loaded {len(filtered_instructions)} instructions")
            
            pref_indices = None
            pref_chunks_map = None
            pref_instructions_map = None
        
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
                    # Pass instructions for EPIC_inst and EPIC_inst_combined
                    if self.method in ["EPIC_inst", "EPIC_inst_combined"]:
                        future = executor.submit(
                            self.process_query,
                            query,
                            preference_text,    # existing cheating method
                            preferences,        # pass all, then infer relevant preference using LLM
                            filtered_chunks,
                            index,  # pass index
                            method_dir,
                            generation_prompt,
                            filtered_instructions,
                            pref_indices,
                            pref_chunks_map,
                            pref_instructions_map
                        )
                    else:
                        future = executor.submit(
                            self.process_query,
                            query,
                            preference_text,    # existing cheating method
                            preferences,        # pass all, then infer relevant preference using LLM
                            filtered_chunks,
                            index,  # pass index
                            method_dir,
                            generation_prompt
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