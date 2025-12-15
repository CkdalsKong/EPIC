CUDA_VISIBLE_DEVICES=0 python EPIC_main.py --method EPIC_inst --persona_index all --mode stream --output_dir stream --dataset PrefWiki --doc_mode wiki --llm_model_name Qwen/Qwen2.5-1.5B-Instruct --vllm_server_url 8000 --stream_batch_size 5000

# 재평가 (모든 stream 실행 완료 후)
# python evaluate_stream_with_different_llm.py --base_dir stream_prefeval/lmsys_sampled/EPIC_inst/ --vllm_url http://localhost:8000 --eval_model meta-llama/Llama-3.1-8B-Instruct