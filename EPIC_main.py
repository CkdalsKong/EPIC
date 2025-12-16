import os
import json
import time
import argparse
import numpy as np
import faiss
from EPIC_utils import EPICUtils
from EPIC_indexing import EPICIndexing
from EPIC_generation import EPICGeneration
from EPIC_evaluation import EPICEvaluation
from EPIC_stream import StreamSetup, StreamManager
import multiprocessing

class EPICMain:
    def __init__(self, mode="all", method="all", device="cuda:0", output_dir="output", dataset="PrefWiki", emb_model_name="facebook/contriever", doc_mode="wiki", vllm_server_url="http://localhost:8008/v1", llm_model_name="meta-llama/Llama-3.1-8B-Instruct", use_local_llm=False, stream_batch_size=2000, stream_num_add=2, stream_num_remove=1, stream_seed=None):
        self.mode = mode
        self.method = method
        self.device = device
        self.output_dir = output_dir

        self.emb_model_name = emb_model_name
        self.doc_mode = doc_mode
        self.vllm_server_url = vllm_server_url
        self.llm_model_name = llm_model_name
        self.use_local_llm = use_local_llm
        
        # Stream mode options
        self.stream_batch_size = stream_batch_size
        self.stream_num_add = stream_num_add
        self.stream_num_remove = stream_num_remove
        self.stream_seed = stream_seed
        
        self.utils = EPICUtils(
            mode=mode,
            method=method,
            device=device,
            output_dir=output_dir,
            dataset=dataset,
            emb_model_name=emb_model_name,
            doc_mode=doc_mode,
            vllm_server_url=vllm_server_url,
            llm_model_name=llm_model_name,
            use_local_llm=use_local_llm,
        )
        
        self.indexing = EPICIndexing(self.utils)
        self.generation = EPICGeneration(self.utils)
        self.evaluation = EPICEvaluation(self.utils)
        self.stream_manager = StreamManager(self.utils)
        self._models_loaded = False
        self._chunks_cache = None
        self._embeddings_cache = None
        self._related_chunks_cache = None
        self._related_embeddings_cache = None
    
    def _load_common_resources(self):
        print("Loading common resources...")
        
        self.utils.load_models()
        
        with open(self.utils.chunk_file, "r", encoding="utf-8") as f:
                self._chunks_cache = [json.loads(line)["text"] for line in f]

        self._embeddings_cache = np.load(self.utils.embedding_file)
        
        self._models_loaded = True
        if self._chunks_cache is not None:  
            print(f"   Chunks: {len(self._chunks_cache)}")
        if self._embeddings_cache is not None:
            print(f"   Embeddings: {self._embeddings_cache.shape}")
    
    def get_cached_resources(self):
        if not self._models_loaded:
            self._load_common_resources()
        
        return {
            "chunks": self._chunks_cache,
            "embeddings": self._embeddings_cache,
            "models_loaded": True
        }
    
    def run_batch_processing(self, persona_indices):
        print(f"\nStarting batch processing for {len(persona_indices)} personas...")
        
        self._load_common_resources()

        for persona_index in persona_indices:
            self.run_single_persona(persona_index)
    
    def run_single_persona(self, persona_index):
        print(f"\n=== Processing persona {persona_index}  ===")
        
        # For standard method: use common directory (mydata style)
        if self.method == "standard":
            if self.llm_model_name == "openai/gpt-oss-20b":
                method_dir = os.path.join(self.utils.output_dir, f"{self.method}_oss")
                data_dir = self.utils.data_dir
            elif self.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
                method_dir = os.path.join(self.utils.output_dir, f"{self.method}_qwen")
                data_dir = self.utils.data_dir
            else:
                method_dir = os.path.join(self.utils.output_dir, f"{self.method}")
                data_dir = self.utils.data_dir
        else:
            if self.llm_model_name == "openai/gpt-oss-20b":
                method_dir = os.path.join(self.utils.output_dir, f"{self.method}_oss/{persona_index}")
                data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
            elif self.llm_model_name == "Qwen/Qwen3-4B-Instruct-2507":
                method_dir = os.path.join(self.utils.output_dir, f"{self.method}_qwen/{persona_index}")
                data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")
            else:
                method_dir = os.path.join(self.utils.output_dir, f"{self.method}/{persona_index}")
                data_dir = os.path.join(self.utils.data_dir, f"{persona_index}")

        print(f"🔍 Data directory: {data_dir}")
        os.makedirs(method_dir, exist_ok=True)
        print(self.utils.output_dir)
        cached_resources = self.get_cached_resources()

        # 1. Indexing
        if self.mode in ["indexing", "all"]:
            print("\n1. Starting indexing...")
            # For standard method: check index in data_dir (mydata style)
            if self.method == "standard":
                faiss_index_path = os.path.join(data_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")
            else:
                faiss_index_path = os.path.join(data_dir, f"index_{self.emb_model_name.replace('/', '_')}.faiss")

            print(f"faiss_index_path: {faiss_index_path}, dataset: {self.utils.dataset_name}")

            if os.path.exists(faiss_index_path):
                print(f"✅ Indexing already completed. Skipping...")
            else:
                self.indexing.run_indexing_with_cache(persona_index, cached_resources)
                print(f"✅ Indexing completed. Results saved to {method_dir}")
        
        # 2. Generation
        if self.mode in ["generation", "all"]:
            print("\n2. Starting generation...")

            gen_file = os.path.join(method_dir, f"gen_{self.method}_flat_{persona_index}.json")

            if os.path.exists(gen_file):
                print(f"✅ Generation already completed. Skipping...")
            else:
                self.generation.run_generation_with_cache(persona_index, method_dir, cached_resources)
                print(f"✅ Generation completed. Results saved to {gen_file}")
        
        # 3. Evaluation
        if self.mode in ["evaluation", "all"]:
            print("\n3. Starting evaluation...")
            eval_file = os.path.join(method_dir, f"eval_{self.method}_flat_{persona_index}.json")
            if os.path.exists(eval_file):
                print(f"✅ Evaluation already completed. Skipping...")
            else:
                self.evaluation.run_evaluation_with_cache(persona_index, method_dir, cached_resources)
                print(f"✅ Evaluation completed. Results saved to {eval_file}")
        
        # 4. Stream mode
        if self.mode == "stream":
            print("\n4. Starting stream processing...")
            print(f"   Batch size: {self.stream_batch_size}")
            
            # Create random preference events avoiding checkpoint timings
            # Use persona_index as part of seed to ensure same persona gets same events
            # but different personas get different events
            total_docs = len(cached_resources["chunks"])
            event_seed = self.stream_seed if self.stream_seed is not None else persona_index
            preference_events = self.stream_manager.create_fixed_preference_events(
                batch_size=self.stream_batch_size,
                total_docs=total_docs,
                num_add=self.stream_num_add,
                num_remove=self.stream_num_remove,
                seed=event_seed
            )
            
            print(f"   Preference events:")
            for event in preference_events:
                print(f"      - {event['type'].upper()} at {event['at_docs']} docs")
            
            # Run stream experiment (skip evaluation by default)
            stream = self.stream_manager.run_stream_experiment(
                persona_index=persona_index,
                all_chunks=cached_resources["chunks"],
                all_embeddings=cached_resources["embeddings"],
                method_dir=method_dir,
                batch_size=self.stream_batch_size,
                preference_events=preference_events,
                skip_evaluation=True,  # Skip evaluation during checkpoint
                stream_seed=event_seed  # Pass seed for preference event handling
            )
            print(f"✅ Stream processing completed. Results saved to stream directory.")
        
        print(f"\n=== Completed persona {persona_index} ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["standard", "cosine", "EPIC", "EPIC_inst", "EPIC_inst_combined", "EPIC_insight", "EPIC_insight_combined"])
    parser.add_argument("--persona_index", type=str, required=True, help="Persona index (0-10) or 'all'")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0)")
    parser.add_argument("--mode", type=str, required=True, choices=["indexing", "generation", "evaluation", "all", "stream"], help="Mode to run: 'indexing', 'generation', 'evaluation', 'all', or 'stream'")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--emb_model_name", type=str, default="facebook/contriever")
    parser.add_argument("--doc_mode", type=str, required=True, choices=["wiki", "eli5", "wiki_total", "eli5_total", "lmsys", "lmsys_sampled"], help="Document mode: 'wiki' for PrefWiki, PrefRQ, 'eli5' for PrefELI5, 'lmsys' for PrefEval (100k docs), 'lmsys_sampled' for PrefEval (10k docs)")
    parser.add_argument("--vllm_server_url", type=str, default="8008", help="vLLM server URL or port number (e.g., 8006 or http://localhost:8008/v1)")
    parser.add_argument("--llm_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="LLM model name")
    parser.add_argument("--use_local_llm", action="store_true", help="Use local LLM inference instead of vLLM server (for Qwen etc.)")
    # Stream mode options
    parser.add_argument("--stream_batch_size", type=int, default=2000, help="Batch size for stream mode (documents per batch)")
    parser.add_argument("--stream_num_add", type=int, default=2, help="Number of preference add events in stream mode")
    parser.add_argument("--stream_num_remove", type=int, default=1, help="Number of preference remove events in stream mode")
    parser.add_argument("--stream_seed", type=int, default=None, help="Random seed for stream preference events (same seed = same events across methods)")
    args = parser.parse_args()

    # vLLM URL Processing
    vllm_server_url = args.vllm_server_url
    if vllm_server_url.isdigit():
        vllm_server_url = f"http://localhost:{vllm_server_url}/v1"
    elif not vllm_server_url.startswith("http"):
        vllm_server_url = f"http://localhost:{vllm_server_url}/v1"
    
    # Persona index setting
    if args.dataset == "PrefWiki":
        indices = list(range(57)) if args.persona_index == "all" else [int(args.persona_index)]
    elif args.dataset == "PrefRQ":
        indices = list(range(90)) if args.persona_index == "all" else [int(args.persona_index)]
    elif args.dataset == "PrefELI5":
        indices = list(range(73)) if args.persona_index == "all" else [int(args.persona_index)]
    elif args.dataset == "PrefEval":
        indices = list(range(57)) if args.persona_index == "all" else [int(args.persona_index)]

    # EPICMain instance creation
    epic = EPICMain(
        mode=args.mode,
        method=args.method,
        device=args.device,
        output_dir=args.output_dir,
        dataset=args.dataset,
        emb_model_name=args.emb_model_name,
        doc_mode=args.doc_mode,
        vllm_server_url=vllm_server_url,
        llm_model_name=args.llm_model_name,
        use_local_llm=args.use_local_llm,
        stream_batch_size=args.stream_batch_size,
        stream_num_add=args.stream_num_add,
        stream_num_remove=args.stream_num_remove,
        stream_seed=args.stream_seed,
    )

    if indices:
        epic.run_batch_processing(indices)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
