import json
from pathlib import Path


def normalize_preference(pref: str) -> str:
    """
    preference 문자열을 정규화합니다.
    - 앞뒤 공백 제거
    - 앞뒤의 따옴표(", ') 제거 (이스케이프된 따옴표 포함)
    """
    pref = pref.strip()
    
    # 앞뒤의 따옴표 제거 (여러 번 반복해서 완전히 제거)
    while pref and pref[0] in ('"', "'", '\\'):
        if pref.startswith('\\"'):
            pref = pref[2:]
        elif pref[0] in ('"', "'"):
            pref = pref[1:]
        else:
            break
    
    while pref and pref[-1] in ('"', "'", '\\'):
        if pref.endswith('\\"'):
            pref = pref[:-2]
        elif pref[-1] in ('"', "'"):
            pref = pref[:-1]
        else:
            break
    
    return pref.strip()


def load_explicit_preferences(explicit_pref_dir: str) -> dict:
    """
    data/explicit_preference 디렉토리의 모든 JSON 파일을 로드하여
    preference -> question 매핑 딕셔너리를 생성합니다.
    """
    pref_to_question = {}
    
    for json_file in Path(explicit_pref_dir).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            preference = item.get("preference", "").strip()
            question = item.get("question", "").strip()
            
            if preference and question:
                # 정규화된 preference를 키로 사용
                normalized_pref = normalize_preference(preference)
                pref_to_question[normalized_pref] = question
    
    return pref_to_question


def create_prefeval(prefeval_path: str, explicit_pref_dir: str, output_path: str):
    """
    기존 PrefEval.json을 읽고, explicit_preference에서 매칭되는 question을 찾아
    새로운 PrefEval.json을 생성합니다.
    """
    # explicit_preference에서 preference -> question 매핑 로드
    pref_to_question = load_explicit_preferences(explicit_pref_dir)
    print(f"Loaded {len(pref_to_question)} preferences from explicit_preference")
    
    # 기존 PrefEval.json 로드
    with open(prefeval_path, "r", encoding="utf-8") as f:
        prefeval_data = json.load(f)
    
    print(f"Loaded {len(prefeval_data)} personas from PrefEval.json")
    
    # 새로운 PrefEval 데이터 생성
    new_prefeval_data = []
    matched_count = 0
    unmatched_prefs = []
    
    for persona in prefeval_data:
        new_persona = {
            "persona_index": persona["persona_index"],
            "preference_blocks": []
        }
        
        for pref_block in persona["preference_blocks"]:
            preference = pref_block["preference"].strip()
            # 정규화된 preference로 매칭
            normalized_pref = normalize_preference(preference)
            
            # explicit_preference에서 매칭되는 question 찾기
            if normalized_pref in pref_to_question:
                matched_question = pref_to_question[normalized_pref]
                matched_count += 1
            else:
                # 매칭되지 않으면 기존 첫 번째 질문 사용
                matched_question = pref_block["queries"][0]["question"]
                unmatched_prefs.append(preference[:50] + "...")
            
            new_pref_block = {
                "preference": normalized_pref,  # 정규화된 preference 사용
                "queries": [
                    {"question": matched_question}
                ]
            }
            new_persona["preference_blocks"].append(new_pref_block)
        
        new_prefeval_data.append(new_persona)
    
    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_prefeval_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults:")
    print(f"  - Total preference blocks processed: {matched_count + len(unmatched_prefs)}")
    print(f"  - Matched from explicit_preference: {matched_count}")
    print(f"  - Unmatched (used first query): {len(unmatched_prefs)}")
    
    if unmatched_prefs:
        print(f"\n  Unmatched preferences (first 5):")
        for pref in unmatched_prefs[:5]:
            print(f"    - {pref}")
    
    print(f"\nSaved to: {output_path}")


def main():
    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent.parent.parent  # PRAG-Baseline
    explicit_pref_dir = base_dir / "data" / "explicit_preference"
    prefeval_path = base_dir / "EPIC" / "dataset" / "PrefEval.json"
    output_path = base_dir / "EPIC" / "dataset" / "PrefEval.json"
    
    print(f"Base directory: {base_dir}")
    print(f"Explicit preference directory: {explicit_pref_dir}")
    print(f"Input PrefEval.json: {prefeval_path}")
    print(f"Output PrefEval.json: {output_path}")
    print()
    
    if not explicit_pref_dir.exists():
        print(f"Error: explicit_preference directory not found: {explicit_pref_dir}")
        return
    
    if not prefeval_path.exists():
        print(f"Error: PrefEval.json not found: {prefeval_path}")
        return
    
    create_prefeval(str(prefeval_path), str(explicit_pref_dir), str(output_path))


if __name__ == "__main__":
    main()

