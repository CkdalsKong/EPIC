#!/usr/bin/env python3
"""
EPIC_inst 메소드별 persona 평균 메모리 계산 스크립트
"""

import os
import json
import argparse
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

# Define the folders and their expected persona counts
FOLDERS = {
    'prefwiki': {
        'output_path': 'output_prefwiki/wiki',
        'data_path': 'data/indexing/wiki',
        'persona_count': 57,
        'doc_mode': 'wiki'
    },
    'prefeval': {
        'output_path': 'output_prefeval/lmsys_sampled',
        'data_path': 'data/indexing/lmsys_sampled',
        'persona_count': 57,
        'doc_mode': 'lmsys_sampled'
    },
    'prefeli5': {
        'output_path': 'output_prefeli5/eli5',
        'data_path': 'data/indexing/eli5',
        'persona_count': 73,
        'doc_mode': 'eli5'
    },
    'prefrq': {
        'output_path': 'output_rq/wiki',
        'data_path': 'data/indexing/wiki',
        'persona_count': 90,
        'doc_mode': 'wiki'
    }
}

# Methods to analyze (EPIC_inst variants)
TARGET_METHODS = ['EPIC_inst', 'EPIC_inst_qwen', 'EPIC_inst_oss']

def get_file_size_mb(file_path: str) -> float:
    """파일 크기를 MB 단위로 반환"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)  # MB
    return 0.0

def get_directory_size_mb(dir_path: str) -> float:
    """디렉토리 전체 크기를 MB 단위로 반환"""
    if not os.path.exists(dir_path):
        return 0.0
    
    total_size = 0
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += get_file_size_mb(file_path)
    except Exception as e:
        print(f"⚠️ Error reading directory {dir_path}: {e}")
        return 0.0
    
    return total_size

def get_llm_suffix(method: str) -> str:
    """메소드명에서 LLM suffix 추출"""
    if method == 'EPIC_inst_oss':
        return '_oss'
    elif method == 'EPIC_inst_qwen':
        return '_qwen'
    else:
        return ''

def get_persona_range(persona_count: int) -> range:
    """persona_count에 따른 persona 범위 반환"""
    return range(persona_count)

def calculate_epic_inst_memory(
    base_dir: Path,
    folder_name: str,
    folder_info: Dict,
    method: str
) -> Tuple[float, int]:
    """
    EPIC_inst 메소드의 메모리 계산
    
    Returns:
        (average_memory_mb, valid_persona_count)
    """
    output_path = base_dir / folder_info['output_path']
    data_path = base_dir / folder_info['data_path']
    persona_count = folder_info['persona_count']
    doc_mode = folder_info['doc_mode']
    
    llm_suffix = get_llm_suffix(method)
    
    # Method name for output directory (EPIC_inst, EPIC_inst_qwen, EPIC_inst_oss)
    output_method_name = method
    
    # Method name for data directory (EPIC_inst_prefwiki, EPIC_inst_prefwiki_oss, etc.)
    if folder_name == 'prefrq':
        data_method_name = f"EPIC_inst_rq{llm_suffix}"
    else:
        data_method_name = f"EPIC_inst_{folder_name}{llm_suffix}"
    
    # Preference embeddings dataset name for file naming
    if folder_name == 'prefrq':
        pref_emb_dataset_name = 'rq'
    else:
        pref_emb_dataset_name = folder_name
    
    persona_range = get_persona_range(persona_count)
    total_memory = 0.0
    valid_personas = 0
    
    print(f"\n  📊 {method} 메모리 계산 중...")
    print(f"    Output 경로: {output_path}/{output_method_name}")
    print(f"    Data 경로: {data_path}/{data_method_name}")
    
    # Preference embeddings are in data/indexing/ directory
    indexing_base_dir = base_dir / "data" / "indexing"
    
    for persona_index in persona_range:
        persona_memory = 0.0
        
        # 1. Output directory files
        output_persona_dir = output_path / output_method_name / str(persona_index)
        
        # instructions.jsonl
        instructions_file = output_persona_dir / "instructions.jsonl"
        instructions_memory = get_file_size_mb(str(instructions_file))
        persona_memory += instructions_memory
        
        # kept.jsonl
        kept_file = output_persona_dir / "kept.jsonl"
        kept_memory = get_file_size_mb(str(kept_file))
        persona_memory += kept_memory
        
        # 2. Data directory files
        data_persona_dir = data_path / data_method_name / str(persona_index)
        
        # embeddings_*.npy files
        embedding_files = list(data_persona_dir.glob("embeddings_*.npy"))
        embedding_memory = sum(get_file_size_mb(str(f)) for f in embedding_files)
        persona_memory += embedding_memory
        
        # index_*.faiss files
        index_files = list(data_persona_dir.glob("index_*.faiss"))
        index_memory = sum(get_file_size_mb(str(f)) for f in index_files)
        persona_memory += index_memory
        
        # 3. Preference embeddings file (in data/indexing/ directory)
        pref_emb_file = indexing_base_dir / f"preference_embeddings_{persona_index}_{pref_emb_dataset_name}_mp.npy"
        pref_emb_memory = get_file_size_mb(str(pref_emb_file))
        persona_memory += pref_emb_memory
        
        if persona_memory > 0:
            total_memory += persona_memory
            valid_personas += 1
            
            if persona_index % 10 == 0:  # 10개마다 진행상황 출력
                print(f"    Persona {persona_index}: {persona_memory:.2f} MB "
                      f"(instructions: {instructions_memory:.2f}, "
                      f"kept: {kept_memory:.2f}, "
                      f"embeddings: {embedding_memory:.2f}, "
                      f"index: {index_memory:.2f}, "
                      f"pref_emb: {pref_emb_memory:.2f})")
    
    avg_memory = total_memory / valid_personas if valid_personas > 0 else 0.0
    
    print(f"  ✅ {method} 결과:")
    print(f"    유효한 Persona 수: {valid_personas}/{persona_count}")
    print(f"    총 메모리: {total_memory:.2f} MB")
    print(f"    평균 메모리: {avg_memory:.2f} MB")
    
    return avg_memory, valid_personas

def main():
    parser = argparse.ArgumentParser(description="EPIC_inst 메소드별 메모리 계산")
    parser.add_argument("--base_dir", type=str, default=".", 
                       help="기본 디렉토리 경로 (기본값: 현재 디렉토리)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    
    print(f"\n{'='*80}")
    print(f"🔍 EPIC_inst 메모리 통계 계산")
    print(f"📁 기본 디렉토리: {base_dir}")
    print(f"{'='*80}")
    
    results = {}
    
    # Process each folder
    for folder_name, folder_info in FOLDERS.items():
        print(f"\n{'='*80}")
        print(f"📂 {folder_name} 폴더 처리 중...")
        print(f"   예상 Persona 수: {folder_info['persona_count']}")
        print(f"{'='*80}")
        
        folder_results = {}
        
        for method in TARGET_METHODS:
            try:
                avg_memory, valid_count = calculate_epic_inst_memory(
                    base_dir, folder_name, folder_info, method
                )
                
                folder_results[method] = {
                    'average_memory_mb': avg_memory,
                    'valid_persona_count': valid_count,
                    'expected_persona_count': folder_info['persona_count']
                }
                
            except Exception as e:
                print(f"❌ {method} 계산 실패: {e}")
                folder_results[method] = None
        
        results[folder_name] = folder_results
    
    # 결과 요약
    print(f"\n{'='*80}")
    print(f"📋 메모리 사용량 요약")
    print(f"{'='*80}")
    
    # Header
    header = f"{'Folder':<15} {'Expected':<10} "
    for method in TARGET_METHODS:
        header += f"{method:>25} "
    print(header)
    print("-" * 100)
    
    # Data rows
    for folder_name, folder_info in FOLDERS.items():
        expected_count = folder_info['persona_count']
        row = f"{folder_name:<15} {expected_count:<10} "
        
        if folder_name in results:
            for method in TARGET_METHODS:
                if results[folder_name][method] is not None:
                    avg_memory = results[folder_name][method]['average_memory_mb']
                    valid_count = results[folder_name][method]['valid_persona_count']
                    row += f"{avg_memory:6.2f}MB ({valid_count:>3}) "
                else:
                    row += f"{'N/A':>25} "
        else:
            row += f"{'N/A':>25} " * len(TARGET_METHODS)
        
        print(row)
    
    print()
    print(f"{'='*80}")
    print(f"📊 전체 평균 (모든 폴더 평균)")
    print(f"{'='*80}")
    print()
    
    # Calculate overall averages
    for method in TARGET_METHODS:
        memories = []
        total_valid = 0
        
        for folder_name in FOLDERS.keys():
            if folder_name in results and results[folder_name][method] is not None:
                method_data = results[folder_name][method]
                memories.append(method_data['average_memory_mb'])
                total_valid += method_data['valid_persona_count']
        
        if memories:
            overall_avg = sum(memories) / len(memories)
            print(f"{method:20s}: {overall_avg:6.2f} MB (simple average across {len(memories)} folders)")
            print(f"{'':20s}  Total valid personas: {total_valid}")
        else:
            print(f"{method:20s}: No data available")
        print()
    
    # 결과를 JSON으로 저장
    output_file = base_dir / "epic_inst_memory_stats.json"
    result = {
        "base_dir": str(base_dir),
        "memory_stats": results
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 결과 저장됨: {output_file}")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {str(e)}")

if __name__ == "__main__":
    main()

