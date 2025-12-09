import json

def count_words(text: str) -> int:
    # 대략적인 길이 측정용 (적당히만 정확하면 됨)
    return len(text.split())

def conversation_to_chunks(conv,
                           max_words=180,   # chunk당 대략 단어 수
                           max_turns=8,     # chunk당 최대 턴 수
                           overlap_turns=2  # chunk 사이 겹치는 턴 수
                           ):
    """
    conv: [{"role": "user"/"assistant", "content": "..."},
           ...]  (하나의 conversation 전체)
    return: [chunk_text1, chunk_text2, ...]
    """
    chunks = []
    cur_msgs = []
    cur_words = 0

    for msg in conv:
        msg_text = f"{msg['role']}: {msg['content']}"
        msg_words = count_words(msg_text)

        # 이 메시지를 추가하면 제한을 넘는 경우 → 여기서 끊기
        if cur_msgs and (
            cur_words + msg_words > max_words or
            len(cur_msgs) >= max_turns
        ):
            # 지금까지 쌓인 걸 하나의 chunk로 저장
            chunks.append("\n".join(cur_msgs))

            # overlap_turns 만큼 이전 메시지를 다음 chunk에 복사
            if overlap_turns > 0:
                cur_msgs = cur_msgs[-overlap_turns:]
                cur_words = sum(count_words(m) for m in cur_msgs)
            else:
                cur_msgs = []
                cur_words = 0

        # 현재 메시지 추가
        cur_msgs.append(msg_text)
        cur_words += msg_words

    # 마지막 남은 것 처리
    if cur_msgs:
        chunks.append("\n".join(cur_msgs))

    return chunks

input_path = "lmsys_chat1m_conversations_structured.jsonl"
output_path_chunks = "lmsys_chat1m_conv_chunks_text.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path_chunks, "w", encoding="utf-8") as fout:

    for line in fin:
        ex = json.loads(line)
        cid = ex["conversation_id"]
        conv = ex["conversation"]
        chunks = conversation_to_chunks(conv)

        for idx, chunk_text in enumerate(chunks):
            record = {
                "conversation_id": cid,
                "chunk_id": idx,
                "model": ex["model"],
                "language": ex["language"],
                "text": chunk_text,
            }
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

print("saved chunked conversations to", output_path_chunks)
