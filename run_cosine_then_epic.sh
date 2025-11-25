#!/bin/bash

# Script to run cosine indexing for personas 0-9, then EPIC for personas 0-9
# Dataset: PrefELI5, doc_mode: eli5_total, vllm_server_url: 8000

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0,1

echo "=========================================="
echo "Step 1: Running cosine indexing for personas 0-9"
echo "=========================================="

# for i in {0..9}; do
#     echo ""
#     echo "Processing persona $i with cosine method (indexing)..."
#     CUDA_VISIBLE_DEVICES=0,1 python EPIC_main.py \
#         --method cosine \
#         --persona_index $i \
#         --mode indexing \
#         --output_dir output \
#         --dataset PrefELI5 \
#         --doc_mode eli5_total \
#         --vllm_server_url 8000
    
#     if [ $? -ne 0 ]; then
#         echo "❌ Error: Cosine indexing failed for persona $i"
#         exit 1
#     fi
#     echo "✅ Completed cosine indexing for persona $i"
# done

echo ""
echo "=========================================="
echo "Step 2: Running EPIC (all) for personas 0-9"
echo "=========================================="

for i in {5..7}; do
    echo ""
    echo "Processing persona $i with EPIC method (all)..."
    CUDA_VISIBLE_DEVICES=0,1 python EPIC_main.py \
        --method EPIC \
        --persona_index $i \
        --mode all \
        --output_dir output \
        --dataset PrefELI5 \
        --doc_mode eli5_total \
        --vllm_server_url 8000
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: EPIC processing failed for persona $i"
        exit 1
    fi
    echo "✅ Completed EPIC processing for persona $i"
done

echo ""
echo "=========================================="
echo "All processing completed successfully!"
echo "=========================================="

