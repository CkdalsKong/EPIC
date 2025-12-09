import json
output_path_chunks = "lmsys_chat1m_conv_chunks_text.jsonl"

cnt = 0
with open(output_path_chunks, "r", encoding="utf-8") as f:
    for _ in f:
        cnt += 1

print("총 chunk 개수:", cnt)