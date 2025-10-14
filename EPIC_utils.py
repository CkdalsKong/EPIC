import os
import re
import csv
import time
import json
import torch
import warnings
import numpy as np
from bs4 import BeautifulSoup
from difflib import get_close_matches
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.nn.parallel import DataParallel
from tqdm.auto import tqdm
import requests
import random

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

class EPICUtils:
    _seed = 0
    set_global_seed(_seed)
    
    def __init__(self, mode, method, device, output_dir, dataset=None, emb_model_name="facebook/contriever", doc_mode="wiki", vllm_server_url="http://localhost:8008/v1", llm_model_name="meta-llama/Llama-3.1-8B-Instruct"):

        self.root_dir = "data"
        self.top_k = 5

        self.prompt_dir = "prompt"

        self.filtering_system = os.path.join(self.prompt_dir, "filtering_systemprompt.txt")
        self.filtering_user = os.path.join(self.prompt_dir, "filtering_userprompt.txt")
        self.rewriting_system = os.path.join(self.prompt_dir, "rewriting_systemprompt.txt")
        self.rewriting_user = os.path.join(self.prompt_dir, "rewriting_userprompt.txt")
        self.generation_prompt = os.path.join(self.prompt_dir, "generation_prompt.txt")

        self.indexing_report_file = "indexing_report.csv"
        self.generation_report_file = "generation_report.csv"
        self.evaluation_report_file = "evaluation_report.csv"
        
        self.error_type_dir = os.path.join("prompt", "error_type")

        self.mode = mode
        self.method = method
        self.doc_mode = doc_mode
        self.threshold = 0.3
        self.device = device
        self.emb_model_name = emb_model_name
        self.vllm_server_url = vllm_server_url
        self.llm_model_name = llm_model_name
        self.batch_size = 16  # Default batch size

        if dataset == "PrefWiki":
            self.dataset = "dataset/PrefWiki.json"
            self.dataset_name = "PrefWiki"
        elif dataset == "PrefRQ":
            self.dataset = "dataset/PrefRQ.json"
            self.dataset_name = "PrefRQ"
        elif dataset == "PrefELI5":
            self.dataset = "dataset/PrefELI5.json"
            self.dataset_name = "PrefELI5"

        print(f"Persona task file: {self.dataset}")
        print(f"LLM model name: {self.llm_model_name}")
        if self.dataset_name == "PrefWiki":
            if self.llm_model_name == "openai/gpt-oss-20b": 
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_prefwiki_oss")
            elif self.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_prefwiki_qwen")
            else:
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_prefwiki")

        elif self.dataset_name == "PrefELI5":
            if self.llm_model_name == "openai/gpt-oss-20b":
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_prefeli5_oss")
            elif self.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_prefeli5_qwen")
            else:
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_prefeli5")

        elif self.dataset_name == "PrefRQ":
            if self.llm_model_name == "openai/gpt-oss-20b":
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_rq_oss")
            elif self.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_rq_qwen")
            else:
                self.data_dir = os.path.join(self.root_dir, f"indexing/{self.doc_mode}/{self.method}_rq")

        self.vllm_server_url = vllm_server_url
        print(f"Using vllm server for {self.llm_model_name}: {self.vllm_server_url}")
        

        model_name_clean = emb_model_name.replace("/", "_")
        self.output_dir = output_dir
        if self.dataset_name == "PrefWiki":
            self.output_dir = f"{self.output_dir}_prefwiki"
        elif self.dataset_name == "PrefELI5":
            self.output_dir = f"{self.output_dir}_prefeli5"
        elif self.dataset_name == "PrefRQ":
            self.output_dir = f"{self.output_dir}_rq"
        else:
            self.output_dir = f"{self.output_dir}"
        
        if doc_mode == "wiki":
            self.output_dir = f"{self.output_dir}/wiki"
        elif doc_mode == "eli5":
            self.output_dir = f"{self.output_dir}/eli5"
        else:
            self.output_dir = f"{self.output_dir}/sample"

        if doc_mode == "wiki":
            self.chunk_file = "sampled_wiki_chunk_10000.jsonl"
            self.embedding_file = f"sampled_wiki_embedding_{model_name_clean}_10000.npy"
        elif doc_mode == "eli5":
            self.chunk_file = "sampled_eli5_chunk_2000.jsonl"
            self.embedding_file = f"sampled_eli5_embedding_{model_name_clean}_2000.npy"
        
        self.emb_tokenizer = None
        self.emb_model = None
        self.hf_emb_tokenizer = None
        self.hf_model = None

    def load_models(self):
        if self.emb_model_name == "nvidia/NV-Embed-v2":
            print("Loading NV-Embed-v2 model...")
            self.emb_model = AutoModel.from_pretrained(
                self.emb_model_name, 
                trust_remote_code=True,
            )
            self.emb_tokenizer = None
        
            if torch.cuda.device_count() > 1:
                self.batch_size = 16 * torch.cuda.device_count()
                print(f"Using {torch.cuda.device_count()} GPUs with batch size {self.batch_size}")
            else:
                self.batch_size = 16
                print(f"Using single GPU with batch size {self.batch_size}")
            return
        else:
            print(f"Loading {self.emb_model_name} model...")
            self.emb_tokenizer = AutoTokenizer.from_pretrained(self.emb_model_name)
            self.emb_model = AutoModel.from_pretrained(self.emb_model_name).eval()

        self.emb_model = self.emb_model.to(self.device)
        print(f"Embedding model loaded on {self.device}")
    
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def embed_texts_mp(self, texts):
        if self.emb_model_name.startswith("nvidia"):
            print(f"Using NVEmbedV2 approach for {self.emb_model_name}...")

            if len(texts) <= self.batch_size:
                params = {
                    "prompts": texts,
                    "max_length": 512,
                    "instruction": "",
                    "batch_size": self.batch_size
                }
                results = self.emb_model.encode(**params)
            else:
                pbar = tqdm(total=len(texts), desc="Generating embeddings")
                results = []

                for i in range(0, len(texts), self.batch_size):
                    batch_chunks = texts[i:i + self.batch_size]
                    params = {
                        "prompts": batch_chunks,
                        "max_length": 512,
                        "instruction": "",
                        "batch_size": self.batch_size
                    }
                    batch_result = self.emb_model.encode(**params)
                    results.append(batch_result)
                    pbar.update(self.batch_size)
                
                pbar.close()
                results = torch.cat(results, dim=0)
            
            if isinstance(results, torch.Tensor):
                results = results.cpu().numpy()
            
            results = (results.T / np.linalg.norm(results, axis=1)).T
            
            return results
        all_embs = []
        batch_size = self.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.emb_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.emb_model(**inputs)
                embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = embeddings.cpu().numpy()
                all_embs.append(embeddings)
        return np.vstack(all_embs)
    
    def embed_query_mp(self, query):
        if self.emb_model_name.startswith("nvidia"):
            print(f"Using NVEmbedV2 approach for {self.emb_model_name}...")

            chunks = [query]
            params = {
                "prompts": chunks,
                "max_length": 512,
                "instruction": "Instruct: Given a question, retrieve relevant documents that best answer the question.\nQuery: ",
                # "instruction": "",
                "batch_size": self.batch_size
            }
            results = self.emb_model.encode(**params)
            
            if isinstance(results, torch.Tensor):
                results = results.cpu().numpy()
            
            results = (results.T / np.linalg.norm(results, axis=1)).T
            
            return results

        inputs = self.emb_tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
            query_emb = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            query_emb = query_emb.cpu().numpy()
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        return query_emb
    
    def generate_message_vllm(self, messages, system_prompt, max_tokens=512, logprob=False):
        headers = {"Content-Type": "application/json"}
        endpoint = f"{self.vllm_server_url}/chat/completions"
        
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend(messages)
        if self.llm_model_name == "openai/gpt-oss-20b":
            max_tokens = 8192
        payload = {
            "model": self.llm_model_name,
            "messages": formatted_messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 0,
            "top_p": 1.0,
            "top_k": -1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False,
            "dtype": "float32"
        }
        
        # Add extra_body only for OSS model
        if self.llm_model_name == "openai/gpt-oss-20b":
            payload["extra_body"] = {"reasoning_effort": "low"}
        if logprob:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        
        for attempt in range(10):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
                if response.status_code != 200:
                    print(f"Error response: {response.text}")
                response.raise_for_status()
                
                result = response.json()
                message = result["choices"][0]["message"]
                content = message.get("content")
                
                if logprob:
                    return content, result["choices"][0]["logprobs"]
                else:
                    return content
            except requests.exceptions.RequestException as e:
                print(f"[Attempt {attempt+1}/5] Request failed: {e}")
                if attempt < 4:  # Wait before retry if not the last attempt
                    # Exponential backoff (1s, 2s, 4s, 8s)
                    wait_time = min(2 ** attempt, 10)
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        raise RuntimeError("Failed to get response from vLLM server after 5 attempts")

    def parse_explanation_and_answer(self, input_string):
        soup = BeautifulSoup(input_string, "html.parser")
        explanation_tag = soup.find("explanation")
        explanation = explanation_tag.text.strip() if explanation_tag else ""
        answer_tag = soup.find("answer")
        answer = answer_tag.text.strip() if answer_tag else ""
        return explanation, answer

    def parse_preference_and_answer(self, input_string):
        soup = BeautifulSoup(input_string, "html.parser")
        preference_tag = soup.find("preference")
        preference = preference_tag.text.strip() if preference_tag else ""
        answer_tag = soup.find("answer")
        answer = answer_tag.text.strip() if answer_tag else ""
        return preference, answer
    
    def load_persona_data(self, persona_index):
        with open(self.dataset, "r", encoding="utf-8") as f:
            personas = json.load(f)
        return next(p for p in personas if p["persona_index"] == persona_index)

    def load_persona_questions(self, file_path, persona_index):
        with open(file_path, "r", encoding="utf-8") as f:
            personas = json.load(f)
        for p in personas:
            if p["persona_index"] == persona_index:
                all_qs = []
                for block in p["preference_blocks"]:
                    pref = block["preference"]
                    for q in block["queries"]:
                        all_qs.append((pref, q["question"]))
                return all_qs
        raise ValueError(f"Persona index {persona_index} not found.")
    
    def extract_preferences_from_response(self, response):
        soup = BeautifulSoup(response, "html.parser")
        preferences = [tag.text.strip() for tag in soup.find_all("preference")]
        return preferences

    def retrieve_top_k_wq_cosine(self, query, preferences, index, chunks, weighted=False):
        top_k = self.top_k
        start_retrieval = time.time()
        query_emb = self.embed_query_mp(query)
        
        # Collect preference embeddings into numpy array
        preference_embs = []
        for preference in preferences:
            preference_emb = self.embed_query_mp(preference)
            preference_embs.append(preference_emb.squeeze(0))  # Convert to (768,) shape
        
        # Stack into numpy array
        preference_embs = np.vstack(preference_embs)  # (num_pref, 768)
        
        # Calculate cosine similarity with np.dot (embeddings already normalized)
        sims = np.dot(preference_embs, query_emb.T).squeeze()  # (num_pref,)
        
        query_emb = self.embed_query_mp(query)
        
        count = 0
        max_sim = -1
        max_i = -1
        for i in range(len(preferences)):
            if sims[i] > max_sim:
                max_sim = sims[i]
                max_i = i

        if count == 0:
            if weighted:
                query_emb += sims[max_i] * self.embed_query_mp(preferences[max_i])
            else:
                query_emb += self.embed_query_mp(preferences[max_i])

        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        D, I = index.search(query_emb, top_k)
        retrieval_time = time.time() - start_retrieval
        return [chunks[i] for i in I[0]], retrieval_time

    def load_prompt_template(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            
    def format_prompt(self, template, preference_text, chunk_text):
        return template.replace("{preference}", preference_text).replace("{chunk}", chunk_text)

    def save_jsonl(self, file_path, items):
        with open(file_path, 'a', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def save_json(self, file_path, data):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_csv(self, file_path, fieldnames, row, write_header=False):
        write_header = write_header or not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def parse_decision_and_reason(self, input_string):
        """Parse decision and reason from LLM response"""
        soup = BeautifulSoup(input_string, "html.parser")
        decision_tag = soup.find("decision")
        reason_tag = soup.find("reason")
        decision = decision_tag.text.strip() if decision_tag else ""
        reason = reason_tag.text.strip() if reason_tag else ""
        return decision, reason

    def parse_decision_and_reason_preference(self, input_string):
        """Parse decision, reason, and preference from LLM response"""
        soup = BeautifulSoup(input_string, "html.parser")
        decision_tag = soup.find("decision")
        reason_tag = soup.find("reason")
        preference_tag = soup.find("relevant_preference")
        decision = decision_tag.text.strip() if decision_tag else ""
        reason = reason_tag.text.strip() if reason_tag else ""
        preference = preference_tag.text.strip() if preference_tag else ""
        
        # Clean preference text
        preference = self.clean_preference_text(preference)
        
        return decision, reason, preference

    def parse_decision_and_reason_preferences(self, input_string):
        """Parse decision, reason, and preferences from LLM response"""
        soup = BeautifulSoup(input_string, "html.parser")
        decision_tag = soup.find("decision")
        reason_tag = soup.find("reason")
        preference_tags = soup.find_all("preference")
        decision = decision_tag.text.strip() if decision_tag else ""
        reason = reason_tag.text.strip() if reason_tag else ""
        preferences = [tag.text.strip() for tag in preference_tags if tag.text.strip()]
        
        # Clean preference text
        preferences = [self.clean_preference_text(pref) for pref in preferences]

        return decision, reason, preferences

    def clean_preference_text(self, preference_text):
        """Extract actual preference content from various preference text formats"""
        if not preference_text:
            return ""
        
        # Convert multi-line to single line
        preference_text = preference_text.strip()
        
        # Numbers only case (e.g., "1, 2" or "5")
        if preference_text.replace(",", "").replace(" ", "").isdigit():
            return preference_text
        
        # Remove "Preference X:" format
        import re
        preference_text = re.sub(r'^Preference\s+\d+:\s*', '', preference_text, flags=re.IGNORECASE)
        
        # Remove leading number and dot (e.g., "1. I prefer...")
        preference_text = re.sub(r'^\d+\.\s*', '', preference_text)
        
        # Remove quotes
        preference_text = preference_text.strip('"\'')
        
        return preference_text.strip()

    def map_preference_numbers_to_text(self, preference_text, preference_list):
        """Map numbered preferences to actual text"""
        if not preference_text or not preference_list:
            return preference_text
        
        import re
        
        # Find numbers and map to actual preference text
        # "1, 2" -> [1, 2] or "5" -> [5]
        numbers = re.findall(r'\d+', preference_text)
        
        if not numbers:
            return preference_text
        
        # Convert numbers to actual preference text
        mapped_preferences = []
        for num in numbers:
            try:
                index = int(num) - 1  # Convert 1-based index to 0-based
                if 0 <= index < len(preference_list):
                    mapped_preferences.append(preference_list[index])
                else:
                    # Use original number if index is out of range
                    mapped_preferences.append(num)
            except ValueError:
                # Use as-is if not a number
                mapped_preferences.append(num)
        
        if mapped_preferences:
            return "; ".join(mapped_preferences)
        else:
            return preference_text

    def parse_rewrite(self, input_string):
        """Parse rewrite from LLM response"""
        soup = BeautifulSoup(input_string, "html.parser")
        if self.method in ["rewrite_emdir", "final", "final_weighted", "final_each", "final_per_pref"]:
            rewrite_tag = soup.find("rewrite")
        else:
            rewrite_tag = soup.find("rewrite")
        if rewrite_tag:
            return rewrite_tag.text.strip()
        else:
            # Fallback handling when rewrite tag is not found
            text = input_string.strip()
            
            # Pattern 1: Starting with "Based on the provided user preferences..."
            if text.startswith("Based on the provided user preferences"):
                # Find actual content after "Document Chunk:" or "Chunk:"
                chunk_match = re.search(r"(?:Document )?Chunk:\s*(.+?)(?:\n\n|$)", text, re.DOTALL)
                if chunk_match:
                    # Return found chunk content (truncated at \n\n)
                    chunk_content = chunk_match.group(1).strip()
                    return chunk_content
                else:
                    # Return original text if "Document Chunk:" or "Chunk:" is not found    
                    return text
            
            # Return full response for other cases (short responses)
            return None
    def process_chunk_rand_prefs(self, idx, chunk_text, preference_text, prompt_template, prompt_template_system=None, preference_list=None, kept_save_info=None):
        # Sort preferences based on cosine similarity if kept_save_info exists
        if kept_save_info and self.method in ["final"]:
            # Sort preferences with similarities
            pref_sim_pairs = list(zip(kept_save_info['relevant_preferences'], kept_save_info['relevant_similarities']))
            # Sort by similarity in ascending order
            pref_sim_pairs.sort(key=lambda x: x[1], reverse=False)
            # Use all relevant preferences 
            sorted_preferences = [pref for pref, sim in pref_sim_pairs]
            preference_text = "\n".join([f"{i+1}. '{p}'" for i, p in enumerate(sorted_preferences)])
        else:
            # Original method: random shuffle
            shuffled_list = preference_list[:]
            random.Random(idx).shuffle(shuffled_list)
            preference_text = "\n".join([f"{i+1}. '{p}'" for i, p in enumerate(shuffled_list)])
        filled_prompt = self.format_prompt(prompt_template, preference_text, chunk_text)
        
        try:
            if prompt_template_system is None:
                llm_response = self.generate_message_vllm(
                    messages=[{"role": "user", "content": filled_prompt}],
                    system_prompt="You are a helpful assistant for indexing document chunks."
                )
                if llm_response is None:
                    print(f"Warning: LLM returned None response - using Filter decision")
                    return {
                        "chunk": chunk_text,
                        "decision": "Filter",
                        "reason": "LLM returned None response",
                        "status": "failed"
                    }
                decision, reason = self.parse_decision_and_reason(llm_response)
            else:
                llm_response = self.generate_message_vllm(
                    messages=[{"role": "user", "content": filled_prompt}],
                    system_prompt=prompt_template_system
                )
                if llm_response is None:
                    print(f"Warning: LLM returned None response - using Filter decision")
                    return {
                        "chunk": chunk_text,
                        "decision": "Filter",
                        "reason": "LLM returned None response",
                        "status": "failed"
                    }
                decision, reason, preferences = self.parse_decision_and_reason_preferences(llm_response)

            # Map numbered preferences to actual text
            if preference_list and preferences:
                for i, preference in enumerate(preferences):
                    preferences[i] = self.map_preference_numbers_to_text(preference, preference_list)

            if decision == "":
                print(f"Warning: Empty decision from LLM response - using Filter decision")
                return {
                    "chunk": chunk_text,
                    "decision": "Filter",
                    "reason": "Empty decision from LLM response",
                    "status": "failed"
                }
            
            # Return result on successful processing
            if prompt_template_system is None:
                return {
                    "chunk": chunk_text,
                    "decision": decision,
                    "reason": reason,
                    "status": "success"
                }
            else:
                return {
                    "chunk": chunk_text,
                    "decision": decision,
                    "reason": reason,
                    "relevant_preference": preferences,
                    "status": "success"
                }
                
        except Exception as e:
            print(f"Failed to process chunk: {e}")
            return {
                "chunk": chunk_text,
                "decision": "Filter",  # Default to filter on failure
                "reason": f"LLM processing failed: {str(e)}",
                "status": "failed"
            }

    def load_existing_results_with_resume(self, result_file):
        """
        Load already processed chunks from existing result file (JSONL format)
        """
        if os.path.exists(result_file):
            try:
                existing_results = []
                with open(result_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            existing_results.append(json.loads(line.strip()))
                return existing_results
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []

    def rewrite_single(self, entry, rewriting_prompt, preference="N/A", rewriting_prompt_system=None):
        # Check if entry is dict or string
        if isinstance(entry, dict):
            original_chunk = entry["chunk"]
            reason = entry.get("reason", "")
            if self.method == "per_pref":
                preference = entry.get("relevant_preference", [])
                preference_text = "\n".join([f"{i+1}. '{p}'" for i, p in enumerate(preference)]) if isinstance(preference, list) else preference
            else:
                preference_text = preference
            
            try:
                if rewriting_prompt_system is None:
                    filled_prompt = rewriting_prompt.replace("{preference}", preference_text).replace("{chunk}", original_chunk).replace("{reason}", reason)
                    llm_response = self.generate_message_vllm(
                        messages=[{"role": "user", "content": filled_prompt}],
                        system_prompt="You are a helpful assistant tasked with rewriting document chunks."
                    )
                else:
                    preferences = entry.get("relevant_preference", [])
                    preference_text = "\n".join([f"{i+1}. '{p}'" for i, p in enumerate(preferences)]) if isinstance(preferences, list) else preferences
                    filled_prompt = rewriting_prompt.replace("{preference}", preference_text).replace("{chunk}", original_chunk).replace("{reason}", reason)
                    llm_response = self.generate_message_vllm(
                        messages=[{"role": "user", "content": filled_prompt}],
                        system_prompt=rewriting_prompt_system
                    )
                
                if llm_response is None:
                    print(f"Warning: LLM returned None response for rewriting - using original chunk")
                        
            except Exception as e:
                print(f"Failed to get rewriting response: {e} - using original chunk")
                llm_response = None
            
            # Extract content from <rewrite> tag
            rewritten_text = self.parse_rewrite(llm_response) if llm_response else None
            if rewritten_text is None:
                return {
                    "rewritten": original_chunk,
                    "original": original_chunk,
                    "reason": reason
                }
            elif rewriting_prompt_system is not None:
                return {
                    "rewritten": rewritten_text,
                    "original": original_chunk,
                    "reason": reason,
                    "relevant_preference": preference_text
                }
            else:
                return {
                    "rewritten": rewritten_text,
                    "original": original_chunk,
                    "reason": reason
                }
        else:
            # When entry is string (from kept_chunks)
            original_chunk = entry
            
            try:
                if preference == "N/A":
                    filled_prompt = rewriting_prompt.replace("{chunk}", original_chunk)
                else:
                    filled_prompt = rewriting_prompt.replace("{preference}", preference).replace("{chunk}", original_chunk)
                
                llm_response = self.generate_message_vllm(
                    messages=[{"role": "user", "content": filled_prompt}],
                    system_prompt="You are a helpful assistant tasked with rewriting document chunks."
                )
                
                if llm_response is None:
                    print(f"Warning: LLM returned None response for string rewriting - using original chunk")
                        
            except Exception as e:
                print(f"Failed to get string rewriting response: {e} - using original chunk")
                llm_response = None
            
            # Extract content from <rewrite> tag
            rewritten_text = self.parse_rewrite(llm_response) if llm_response else None
            if rewritten_text is None:
                return {
                    "rewritten": original_chunk,
                    "original": original_chunk
                }
            else:
                return {
                    "rewritten": rewritten_text,
                    "original": original_chunk
                }

    def parse_numbered_preferences(self, preference_text, preference_list):
        """
        Parse numbered preference text and match with actual preference text
        
        Args:
            preference_text: Numbered preference text (e.g., "1. I am fascinated by Renaissance...\n5. I love visiting heritage sites...")
            preference_list: Original preference text list
            
        Returns:
            matched_preferences: List of matched preference texts
        """
        matched_preferences = []
        
        # Split by lines
        lines = preference_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if numbered format (e.g., "1. ", "2. ", "5. ")
            if line[0].isdigit() and '. ' in line:
                # Separate number and text
                dot_index = line.find('. ')
                if dot_index != -1:
                    extracted_text = line[dot_index + 2:].strip()
                    
                    # Find extracted text in preference_list
                    for pref in preference_list:
                        if extracted_text == pref:
                            matched_preferences.append(pref)
                            break
                    else:
                        # Check partial match if no exact match
                        for pref in preference_list:
                            if extracted_text in pref or pref in extracted_text:
                                matched_preferences.append(pref)
                                break
                        else:
                            print(f"Warning: Could not match preference text: '{extracted_text}'")
            else:
                # Direct matching when no number
                for pref in preference_list:
                    if line == pref:
                        matched_preferences.append(pref)
                        break
                else:
                    # Check partial match
                    for pref in preference_list:
                        if line in pref or pref in line:
                            matched_preferences.append(pref)
                            break
                    else:
                        print(f"Warning: Could not match preference text: '{line}'")
        
        return matched_preferences

    def get_closest_preference(self, preference, original_preference_list):
        """
        Find most similar preference text based on difflib's get_close_matches
        """
        pref_list_lower = [p.lower() for p in original_preference_list]
        lower_to_orig = {p.lower(): p for p in original_preference_list}
        q = preference.lower().strip()
        best = get_close_matches(q, pref_list_lower, n=1, cutoff=0.0)
        return lower_to_orig[best[0]] if best else None