import os
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class EPICIndexing:
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
    
    def run_indexing_with_cache(self, persona_index, cached_resources):
        if self.utils.llm_model_name == "openai/gpt-oss-20b":
            llm_name = "_oss"
        elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            llm_name = "_qwen"
        else:
            llm_name = ""

        print(f"\n=== Starting indexing for persona {persona_index} ===")

        if self.utils.llm_model_name == "openai/gpt-oss-20b":
            method_dir = os.path.join(self.output_dir, f"{self.method}_oss/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
            method_dir = os.path.join(self.output_dir, f"{self.method}_qwen/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        else:
            method_dir = os.path.join(self.output_dir, f"{self.method}/{persona_index}")
            data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
        
        
        embeddings_file = os.path.join(data_dir, f"embeddings_{self.emb_model_name.replace('/', '_')}.npy")
        index_file = os.path.join(data_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")

        os.makedirs(method_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        print(f"Output directory: {method_dir}")

        print("✅ Using cached resources (models, chunks, embeddings)")
        chunks = cached_resources["chunks"]
        chunk_embeddings = cached_resources["embeddings"]
        print(f"Using {len(chunks)} cached chunks and their embeddings")

        persona = self.utils.load_persona_data(persona_index)
        print(f"Loaded persona data for index {persona_index}")

        print("Loading or generating preference embeddings...")
        preference_list = [block["preference"] for block in persona["preference_blocks"]]
        preference_emb_file_prefix = "nv_" if self.emb_model_name == "nvidia/NV-Embed-v2" else ""
        if self.utils.dataset_name == "PrefWiki":
            preference_emb_file = os.path.join(self.utils.root_dir, f"indexing/{preference_emb_file_prefix}preference_embeddings_{persona_index}_prefwiki_mp.npy")
        elif self.utils.dataset_name == "PrefELI5":
            preference_emb_file = os.path.join(self.utils.root_dir, f"indexing/{preference_emb_file_prefix}preference_embeddings_{persona_index}_prefeli5_mp.npy")
        elif self.utils.dataset_name == "PrefRQ":
            preference_emb_file = os.path.join(self.utils.root_dir, f"indexing/{preference_emb_file_prefix}preference_embeddings_{persona_index}_rq_mp.npy")


        if os.path.exists(preference_emb_file):
            print("Loading existing preference embeddings...")
            preference_embeddings = np.load(preference_emb_file)
        else:
            print("Generating new preference embeddings...")
            preference_embeddings = self.utils.embed_texts_mp(preference_list)\
            
            preference_embeddings = preference_embeddings / np.linalg.norm(preference_embeddings, axis=1, keepdims=True)
            np.save(preference_emb_file, preference_embeddings)
            print(f"Saved preference embeddings to {preference_emb_file}")
            
            
        chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        print(f"Using {len(preference_list)} preference embeddings")
        
        start_total = time.time()

        print(self.utils.threshold)
        
        print("\nStarting cosine similarity filtering...")
        kept_save, kept_chunks, filtered_save = [], [], []
        keep_indices = []
        relevant_preferences = []
        relevant_similarities = []

        cosine_filtering_file = os.path.join(method_dir, "cosine_filtering_results.jsonl")

        # For EPIC method: load cosine filtering results from cosine method's output
        if self.method == "EPIC":
            # Find cosine method's output directory path
            if self.utils.llm_model_name == "openai/gpt-oss-20b":
                cosine_method_dir = os.path.join(self.output_dir, f"cosine_oss/{persona_index}")
            elif self.utils.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
                cosine_method_dir = os.path.join(self.output_dir, f"cosine_qwen/{persona_index}")
            else:
                cosine_method_dir = os.path.join(self.output_dir, f"cosine/{persona_index}")
            
            cosine_filtering_file = os.path.join(cosine_method_dir, "cosine_filtering_results.jsonl")
            
            if os.path.exists(cosine_filtering_file):
                print(f"✅ Found cosine filtering results from cosine method: {cosine_filtering_file}")
                # Load JSONL
                with open(cosine_filtering_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        kept_save.append(entry)
                        kept_chunks.append(entry.get("chunk", ""))
                        relevant_preferences.append(entry.get("relevant_preferences", []))
                        relevant_similarities.append(entry.get("relevant_similarities", []))
                filter_time = 0.0
                print(f"Loaded {len(kept_chunks)} kept chunks from cosine method results")
            else:
                raise FileNotFoundError(f"❌ Cosine filtering results not found. Please run cosine method first: {cosine_filtering_file}")
        
        # For cosine method: only perform cosine filtering and exit
        elif self.method == "cosine":
            if os.path.exists(cosine_filtering_file):
                print(f"⚠️ Found existing cosine results. Skipping filtering: {cosine_filtering_file}")
                # Load JSONL
                with open(cosine_filtering_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        kept_save.append(entry)
                        kept_chunks.append(entry.get("chunk", ""))
                        relevant_preferences.append(entry.get("relevant_preferences", []))
                        relevant_similarities.append(entry.get("relevant_similarities", []))
                filter_time = 0.0
                print(f"Loaded {len(kept_chunks)} kept chunks from cache")
            else:
                batch_size = self.batch_size
                for i in tqdm(range(0, len(chunk_embeddings_norm), batch_size), desc=f"Filtering persona {persona_index}"):
                    batch_embeddings = chunk_embeddings_norm[i:i + batch_size]
                    batch_chunks = chunks[i:i + batch_size]
                    
                    sims = np.dot(preference_embeddings, batch_embeddings.T)
                    above = sims > self.utils.threshold
                    mask = np.any(above, axis=0)
                    
                    for j, (chunk, is_kept) in enumerate(zip(batch_chunks, mask)):
                        if is_kept:
                            kept_chunks.append(chunk)
                            relevant_idx = np.where(above[:, j])[0]
                            relevant_prefs = [preference_list[k] for k in relevant_idx]
                            relevant_sim_values = [sims[k, j] for k in relevant_idx]
                            
                            kept_save.append({
                                "chunk": chunk,
                                "relevant_preferences": relevant_prefs,
                                "relevant_similarities": [float(sim) for sim in relevant_sim_values]
                            })
                            relevant_preferences.append(relevant_prefs)
                            relevant_similarities.append(relevant_sim_values)
                        else:
                            filtered_save.append({"chunk": chunk})
                
                filter_time = time.time() - start_total
                print(f"Cosine filtering completed. Kept {len(kept_chunks)} chunks out of {len(chunks)}")
                self.utils.save_jsonl(cosine_filtering_file, kept_save)
                print(f"✅ Cosine filtering results saved to {cosine_filtering_file}")
            
            # For cosine method: exit after saving JSONL (no CSV report, no LLM filtering, no FAISS indexing)
            total_time = time.time() - start_total
            print(f"\n=== Completed cosine filtering for persona {persona_index} ===")
            print(f"Total time: {total_time:.2f} seconds")
            return method_dir
        
        # For EPIC method: continue with existing logic (LLM filtering, rewriting, FAISS indexing)
        else:
            if os.path.exists(cosine_filtering_file):
                print(f"⚠️ Found existing cosine results. Skipping filtering: {cosine_filtering_file}")
                # Load JSONL
                with open(cosine_filtering_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        kept_save.append(entry)
                        kept_chunks.append(entry.get("chunk", ""))
                        relevant_preferences.append(entry.get("relevant_preferences", []))
                        relevant_similarities.append(entry.get("relevant_similarities", []))
                filter_time = 0.0
                print(f"Loaded {len(kept_chunks)} kept chunks from cache")
            else:
                batch_size = self.batch_size
                for i in tqdm(range(0, len(chunk_embeddings_norm), batch_size), desc=f"Filtering persona {persona_index}"):
                    batch_embeddings = chunk_embeddings_norm[i:i + batch_size]
                    batch_chunks = chunks[i:i + batch_size]
                    
                    sims = np.dot(preference_embeddings, batch_embeddings.T)
                    above = sims > self.utils.threshold
                    mask = np.any(above, axis=0)
                    
                    for j, (chunk, is_kept) in enumerate(zip(batch_chunks, mask)):
                        if is_kept:
                            kept_chunks.append(chunk)
                            relevant_idx = np.where(above[:, j])[0]
                            relevant_prefs = [preference_list[k] for k in relevant_idx]
                            relevant_sim_values = [sims[k, j] for k in relevant_idx]
                            
                            kept_save.append({
                                "chunk": chunk,
                                "relevant_preferences": relevant_prefs,
                                "relevant_similarities": [float(sim) for sim in relevant_sim_values]
                            })
                            relevant_preferences.append(relevant_prefs)
                            relevant_similarities.append(relevant_sim_values)
                        else:
                            filtered_save.append({"chunk": chunk})
                
                filter_time = time.time() - start_total
                print(f"Cosine filtering completed. Kept {len(kept_chunks)} chunks out of {len(chunks)}")
                self.utils.save_jsonl(cosine_filtering_file, kept_save)
                print(f"✅ Cosine filtering results saved to {cosine_filtering_file}")

      
        print("\nStarting LLM filtering...")

        result_info_file = os.path.join(method_dir, "result_info.jsonl")
        rewritten_file = os.path.join(method_dir, "rewritten.jsonl")

        filtering_prompt_system = self.utils.load_prompt_template(self.utils.filtering_system)
        filtering_prompt_user = self.utils.load_prompt_template(self.utils.filtering_user)

        preference_text = "\n".join([f"{i+1}. '{p}'" for i, p in enumerate(preference_list)])

        start_llm = time.time()
        results = []

        if os.path.exists(result_info_file):
            print(f"⚠️ Found existing result info. Skipping LLM filtering: {result_info_file}")
            # result_info was saved via save_json (full JSON array), not JSONL
            try:
                with open(result_info_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except Exception:
                # fallback to JSONL one-per-line
                with open(result_info_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        results.append(json.loads(line))
            llm_time = 0.0
        else:
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.utils.process_chunk_rand_prefs, idx, kept_chunks[idx], preference_text, filtering_prompt_user, filtering_prompt_system, relevant_preferences[idx], kept_save[idx]): idx for idx in range(len(kept_chunks))}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM: persona {persona_index}", leave=False, ncols=100):
                    result = future.result()
                    if result:
                        results.append(result)

                self.utils.save_json(result_info_file, results)
                print(f"✅ Result info saved to {result_info_file}")
                
                llm_time = time.time() - start_llm

        filtered, rewritten, kept = [], [], []
        failed_chunks = [] 
        success_count = 0
        failed_count = 0

        for item in results:
            if item["status"] == "failed":
                failed_chunks.append(item)
                failed_count += 1
                filtered.append({"chunk": item["chunk"]})
            else:
                success_count += 1
                if item["decision"] == "Discard":
                    filtered.append({"chunk": item["chunk"]})
                elif item["decision"] == "Rewrite":
                        rewritten.append({
                            "chunk": item["chunk"],
                            "reason": item["reason"],
                            "relevant_preference": item["relevant_preference"]
                        })
        
        print(f"LLM filtering completed. Success: {success_count}, Failed: {failed_count}")

        print(f"Results - Filtered: {len(filtered)}, rewritten: {len(rewritten)}")

        if failed_chunks:
            failed_file = os.path.join(method_dir, "failed_chunks.jsonl")
            self.utils.save_jsonl(failed_file, failed_chunks)
            print(f"⚠️ Failed chunks saved to {failed_file}")

        preference_text = "\n".join([f"- {p}" for p in preference_list])
        rewriting_prompt_system = self.utils.load_prompt_template(self.utils.rewriting_system)
        rewriting_prompt_user = self.utils.load_prompt_template(self.utils.rewriting_user)
        
        start_rewriting = time.time()
        rewritten_final = []

        if os.path.exists(rewritten_file):
            print(f"⚠️ Found existing rewritten file. Skipping rewriting: {rewritten_file}")
            with open(rewritten_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    rewritten_final.append(json.loads(line))
            rewriting_time = 0.0
        else:
            if rewritten:
                first_entry = rewritten[0]
                first_chunk = first_entry["chunk"]
                preference_text_rew = "\n".join([f"- {p}" for p in preference_list])

                filled_user_prompt = rewriting_prompt_user.format(preference=preference_text_rew, chunk=first_chunk, reason=first_entry["reason"])
                full_prompt = {
                    "system_prompt": rewriting_prompt_system,
                    "user_prompt": filled_user_prompt,
                    "full_conversation": f"System: {rewriting_prompt_system}\n\nUser: {filled_user_prompt}"
                }
                prompt_file = os.path.join(method_dir, "rewriting_prompt_sample.json")
                self.utils.save_json(prompt_file, full_prompt)
                print(f"✅ Rewriting prompt sample saved to {prompt_file}")
            
            with ThreadPoolExecutor() as executor:  
                futures = [executor.submit(self.utils.rewrite_single, entry, rewriting_prompt_user, rewriting_prompt_system=rewriting_prompt_system) for entry in rewritten]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Rewriting chunks", leave=False, ncols=100):
                    try:
                        result = future.result()
                        rewritten_final.append(result)
                    except Exception as e:
                        print(f"Rewriting failed: {e}")
            rewriting_time = time.time() - start_rewriting
            self.utils.save_jsonl(rewritten_file, rewritten_final)
            print(f"✅ Rewriting info saved to {rewritten_file}")
        
        print(f"Rewriting completed. rewritten {len(rewritten_final)} chunks")
        merged_chunks = [item["rewritten"] for item in rewritten_final] + [item["chunk"] for item in kept]
    
        print("\nCreating FAISS index...")
        start_faiss = time.time()
        
        print("Generating embeddings...")

        embeddings = self.utils.embed_texts_mp(merged_chunks)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        print(f"Generated {len(embeddings)} embeddings")
            
        dim = embeddings.shape[1]
        print("Creating FAISS IndexFlatIP...")
        index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity

        index.add(embeddings.astype(np.float32))
        
        print("Saving results...")
        faiss.write_index(index, index_file)
        print(f"FAISS saved in {index_file}")
        np.save(embeddings_file, embeddings)
        self.utils.save_jsonl(os.path.join(method_dir, "kept.jsonl"), [{"text": chunk} for chunk in merged_chunks])
        
        faiss_time = time.time() - start_faiss
        total_time = time.time() - start_total
        
        print("\nGenerating report...")
        fieldnames = ["method", "persona_index", "cosine_kept", "random_kept", "cluster_kept", "llm_filtered", "rewritten", "kept", "cosine_filter_time(s)", "random_filter_time(s)", "cluster_filter_time(s)", "llm_time(s)", "rewriting_time(s)", "faiss_time(s)", "total_time(s)"]
        row = {
            "method": f"{self.method}{llm_name}",
            "persona_index": f"{persona_index}",
            "cosine_kept": len(kept_chunks),
            "random_kept": len(kept_chunks),
            "cluster_kept": 0,
            "llm_filtered": len(filtered),
            "rewritten": len(rewritten_final),
            "kept": len(kept),
            "cosine_filter_time(s)": f"{filter_time:.2f}",
            "random_filter_time(s)": f"{filter_time:.2f}",
            "cluster_filter_time(s)": "0",
            "llm_time(s)": f"{llm_time:.2f}",
            "rewriting_time(s)": f"{rewriting_time:.2f}",
            "faiss_time(s)": f"{faiss_time:.2f}",
            "total_time(s)": f"{total_time:.2f}"
        }
        self.utils.save_csv(os.path.join(self.output_dir, self.utils.indexing_report_file), fieldnames, row)
        
        print(f"\n=== Completed indexing for persona {persona_index} ===")
        print(f"Total time: {total_time:.2f} seconds")
        return method_dir
