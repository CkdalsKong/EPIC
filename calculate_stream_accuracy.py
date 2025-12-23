#!/usr/bin/env python3
"""
Stream 재평가 결과에서 standard, cosine, EPIC_inst 방법의 
체크포인트별 및 전체 평균 정확도를 계산하는 스크립트
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not installed. Plotting will be disabled.")

# 분석할 방법들
TARGET_METHODS = ['standard', 'cosine', 'EPIC_inst']

# Persona 범위 (prefwiki는 0-56)
PERSONA_RANGE = range(57)  # 0-56

def find_stream_dirs(base_dir: Path, method: List[str], persona_index: int) -> List[Path]:
    """특정 방법과 persona_index에 대한 stream 디렉토리들을 찾기 (우선순위에 따라 하나만 선택)"""
    # method 리스트는 이미 우선순위 순서로 정렬되어 있음 (qwen > 기본 > oss)
    # 우선순위에 따라 첫 번째로 존재하는 디렉토리 사용
    for method_name in method:
        # base_dir 구조 확인
        # 경우 1: base_dir이 이미 method 디렉토리 (예: stream_prefwiki/wiki/EPIC_inst/)
        if base_dir.name == method_name:
            method_dir = base_dir / str(persona_index)
        # 경우 2: base_dir이 method 디렉토리의 부모 (예: stream_prefwiki/wiki/)
        elif (base_dir / method_name).exists():
            method_dir = base_dir / method_name / str(persona_index)
        # 경우 3: base_dir이 루트 디렉토리 (예: . 또는 workspace root)
        else:
            method_dir = base_dir / "stream_prefwiki" / "wiki" / method_name / str(persona_index)
        
        if method_dir.exists():
            # stream_* 디렉토리들 찾기
            stream_dirs = [d for d in method_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('stream_')]
            if stream_dirs:
                return stream_dirs
    
    return []

def load_reevaluated_results(stream_dir: Path) -> List[Dict]:
    """재평가 결과 JSON 파일 로드 (재평가 파일이 없으면 원본 체크포인트 결과 사용)"""
    # 먼저 재평가 결과 파일 확인
    results_file = stream_dir / "all_checkpoints_reevaluated.json"
    
    # 재평가 파일이 없으면 원본 체크포인트 결과 사용
    if not results_file.exists():
        # 원본 체크포인트 결과 파일 확인
        results_file = stream_dir / "all_checkpoints.json"
        if not results_file.exists():
            return []
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 각 체크포인트의 metrics.json에서 메모리 정보 추가
        for result in results:
            checkpoint_id = result.get('checkpoint_id')
            if checkpoint_id is not None:
                metrics_file = stream_dir / f"checkpoint_{checkpoint_id}" / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r', encoding='utf-8') as mf:
                            metrics = json.load(mf)
                            # metrics.json에서 memory_mb 가져오기
                            if 'memory_mb' in metrics:
                                result['memory_mb'] = metrics['memory_mb']
                    except Exception as e:
                        # metrics.json 로드 실패해도 계속 진행
                        pass
        
        return results
    except Exception as e:
        print(f"⚠️ Error loading {results_file}: {e}")
        return []

def load_preference_events(stream_dir: Path) -> List[Dict]:
    """Preference events 로드"""
    meta_file = stream_dir / "stream_metadata.json"
    if not meta_file.exists():
        print(f"  ⚠️ stream_metadata.json not found at: {meta_file}")
        return []
    
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            events = metadata.get("preference_events", [])
            print(f"  ✅ Loaded {len(events)} preference events from {meta_file}")
            return events
    except Exception as e:
        print(f"  ⚠️ Error loading preference events from {meta_file}: {e}")
        return []

def calculate_checkpoint_accuracy(
    base_dir: Path,
    method: List[str]
) -> Tuple[Dict[int, List[float]], List[float], Dict[int, List[float]], Dict[int, List[Dict]], Dict[int, Dict[int, int]]]:
    """
    특정 방법에 대한 체크포인트별 정확도 및 메모리 수집
    
    Returns:
        (checkpoint_accuracies, all_accuracies, checkpoint_memories, preference_events_by_persona, checkpoint_to_docs_by_persona)
        - checkpoint_accuracies: {checkpoint_id: [accuracy1, accuracy2, ...]}
        - all_accuracies: 모든 체크포인트의 모든 정확도 리스트
        - checkpoint_memories: {checkpoint_id: [memory1, memory2, ...]}
        - preference_events_by_persona: {persona_index: [event1, event2, ...]}
        - checkpoint_to_docs_by_persona: {persona_index: {checkpoint_id: docs_processed}}
    """
    checkpoint_accuracies = defaultdict(list)
    all_accuracies = []
    checkpoint_memories = defaultdict(list)
    preference_events_by_persona = {}
    checkpoint_to_docs_by_persona = {}
    
    print(f"\n{'='*80}")
    print(f"📊 {', '.join(method)} 방법 분석 중...")
    print(f"{'='*80}")
    
    valid_personas = 0
    
    for persona_index in PERSONA_RANGE:
        stream_dirs = find_stream_dirs(base_dir, method, persona_index)
        
        if not stream_dirs:
            # 디버깅: 첫 번째 persona만 상세 로그 출력
            if persona_index == PERSONA_RANGE[0]:
                print(f"  ⚠️ Persona {persona_index}: No stream directories found")
                print(f"     Base dir: {base_dir}")
                print(f"     Methods tried: {method}")
                for method_name in method:
                    if (base_dir / method_name).exists():
                        method_dir = base_dir / method_name / str(persona_index)
                        print(f"     - {method_name}: {method_dir} exists={method_dir.exists()}")
                        if method_dir.exists():
                            subdirs = list(method_dir.iterdir())
                            print(f"       Subdirs: {[d.name for d in subdirs if d.is_dir()]}")
            continue
        
        # 가장 최근 stream 디렉토리 사용 (우선순위에 따라 선택된 디렉토리)
        stream_dir = max(stream_dirs, key=lambda x: x.stat().st_mtime)
        
        results = load_reevaluated_results(stream_dir)
        
        if not results:
            continue
        
        valid_personas += 1
        persona_checkpoints = {}
        checkpoint_to_docs = {}
        
        # Load preference events for this persona
        print(f"  🔍 Loading preference events for persona {persona_index} from: {stream_dir}")
        events = load_preference_events(stream_dir)
        if events:
            preference_events_by_persona[persona_index] = events
            print(f"  ✅ Stored {len(events)} events for persona {persona_index}")
        else:
            print(f"  ⚠️ No preference events found for persona {persona_index}")
        
        for checkpoint in results:
            checkpoint_id = checkpoint.get('checkpoint_id')
            accuracy = checkpoint.get('preference_following_accuracy')
            memory_mb = checkpoint.get('memory_mb', 0)
            docs_processed = checkpoint.get('docs_processed', 0)
            
            if checkpoint_id is not None and accuracy is not None:
                checkpoint_accuracies[checkpoint_id].append(accuracy)
                all_accuracies.append(accuracy)
                persona_checkpoints[checkpoint_id] = accuracy
                
                # Collect memory information
                if memory_mb > 0:
                    checkpoint_memories[checkpoint_id].append(memory_mb)
                
                # Store docs_processed for checkpoint_id mapping
                if docs_processed > 0:
                    checkpoint_to_docs[checkpoint_id] = docs_processed
        
        if checkpoint_to_docs:
            checkpoint_to_docs_by_persona[persona_index] = checkpoint_to_docs
        
        if persona_index % 10 == 0:
            print(f"  Persona {persona_index}: {len(results)} checkpoints found")
    
    print(f"  ✅ 유효한 Persona 수: {valid_personas}/{len(PERSONA_RANGE)}")
    
    return checkpoint_accuracies, all_accuracies, checkpoint_memories, preference_events_by_persona, checkpoint_to_docs_by_persona

def calculate_averages(checkpoint_accuracies: Dict[int, List[float]]) -> Dict[int, float]:
    """체크포인트별 평균 정확도 계산"""
    averages = {}
    for checkpoint_id, accuracies in sorted(checkpoint_accuracies.items()):
        if accuracies:
            averages[checkpoint_id] = sum(accuracies) / len(accuracies)
    return averages

def main():
    parser = argparse.ArgumentParser(description="Stream 재평가 결과 분석")
    parser.add_argument("--base_dir", type=str, default=".", 
                       help="기본 디렉토리 경로 (기본값: 현재 디렉토리)")
    parser.add_argument("--persona_index", type=str, default=None,
                       help="평가할 persona 인덱스 (예: '0', '0,1,2', 'all' - 기본값: all)")
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    
    # Persona 범위 결정
    global PERSONA_RANGE
    if args.persona_index is None or args.persona_index.lower() == 'all':
        PERSONA_RANGE = range(57)  # 0-56
        print(f"📊 모든 persona 평가 (0-56)")
    else:
        # 콤마로 구분된 인덱스 파싱
        try:
            indices = [int(x.strip()) for x in args.persona_index.split(',')]
            PERSONA_RANGE = sorted(set(indices))
            print(f"📊 선택된 persona 평가: {PERSONA_RANGE}")
        except ValueError:
            print(f"❌ 잘못된 persona_index 형식: {args.persona_index}")
            print(f"   예시: '0', '0,1,2', 'all'")
            return
    
    print(f"\n{'='*80}")
    print(f"🔍 Stream 재평가 결과 분석")
    print(f"📁 기본 디렉토리: {base_dir}")
    print(f"{'='*80}")
    
    all_results = {}
    
    # 각 방법별로 분석
    for method_name in TARGET_METHODS:
        # 방법 이름에 따른 검색 리스트 생성 (우선순위: qwen > 기본 > oss)
        if method_name == 'standard':
            method_list = ['standard_qwen', 'standard', 'standard_oss']
        elif method_name == 'cosine':
            method_list = ['cosine_qwen', 'cosine', 'cosine_oss']
        elif method_name == 'EPIC_inst':
            method_list = ['EPIC_inst_qwen', 'EPIC_inst', 'EPIC_inst_oss']
        else:
            method_list = [method_name]
        
        checkpoint_accuracies, all_accuracies, checkpoint_memories, preference_events_by_persona, checkpoint_to_docs_by_persona = calculate_checkpoint_accuracy(
            base_dir, method_list
        )
        
        # 체크포인트별 평균 계산
        checkpoint_averages = calculate_averages(checkpoint_accuracies)
        checkpoint_memory_averages = calculate_averages(checkpoint_memories)
        
        # 전체 평균 계산
        overall_avg = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        overall_memory = sum([sum(memories) for memories in checkpoint_memories.values()]) / sum([len(memories) for memories in checkpoint_memories.values()]) if checkpoint_memories else 0.0
        
        all_results[method_name] = {
            'checkpoint_averages': checkpoint_averages,
            'checkpoint_memory_averages': checkpoint_memory_averages,
            'overall_average': overall_avg,
            'overall_memory': overall_memory,
            'total_checkpoints': len(all_accuracies),
            'unique_checkpoint_ids': sorted(checkpoint_averages.keys()),
            'preference_events_by_persona': preference_events_by_persona,
            'checkpoint_to_docs_by_persona': checkpoint_to_docs_by_persona
        }
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"📋 체크포인트별 평균 정확도")
    print(f"{'='*80}")
    
    # 모든 방법에서 나타나는 체크포인트 ID 수집
    all_checkpoint_ids = set()
    for method_name in TARGET_METHODS:
        all_checkpoint_ids.update(all_results[method_name]['unique_checkpoint_ids'])
    all_checkpoint_ids = sorted(all_checkpoint_ids)
    
    # 헤더
    header = f"{'Checkpoint':<12} "
    for method_name in TARGET_METHODS:
        header += f"{method_name:>15} "
    print(header)
    print("-" * 60)
    
    # 체크포인트별 결과
    for checkpoint_id in all_checkpoint_ids:
        row = f"{checkpoint_id:<12} "
        for method_name in TARGET_METHODS:
            avg = all_results[method_name]['checkpoint_averages'].get(checkpoint_id, None)
            if avg is not None:
                row += f"{avg:>15.2f}% "
            else:
                row += f"{'N/A':>15} "
        print(row)
    
    # 전체 평균 출력
    print(f"\n{'='*80}")
    print(f"📊 전체 평균 정확도")
    print(f"{'='*80}")
    print()
    
    # Persona가 하나일 때 정보 출력
    if len(PERSONA_RANGE) == 1:
        persona_idx = PERSONA_RANGE[0]
        print(f"📌 분석된 Persona: {persona_idx}")
        print()
        
        # 각 방법별 preference events 출력
        for method_name in TARGET_METHODS:
            events_by_persona = all_results[method_name].get('preference_events_by_persona', {})
            if persona_idx in events_by_persona:
                events = events_by_persona[persona_idx]
                if events:
                    print(f"  {method_name} - Preference Events:")
                    for event in events:
                        event_type = event.get('type', 'unknown')
                        docs_processed = event.get('docs_processed', 0)
                        pref_text = event.get('preference', '')
                        print(f"    - {event_type.upper()} at {docs_processed} docs: {pref_text[:60]}...")
        print()
    
    for method_name in TARGET_METHODS:
        overall_avg = all_results[method_name]['overall_average']
        overall_memory = all_results[method_name]['overall_memory']
        total_checkpoints = all_results[method_name]['total_checkpoints']
        print(f"{method_name:15s}: {overall_avg:6.2f}% | 메모리: {overall_memory:6.2f} MB (총 {total_checkpoints} 체크포인트)")
    
    # 결과를 JSON으로 저장
    output_file = base_dir / "stream_accuracy_summary.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 결과 저장됨: {output_file}")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {str(e)}")
    
    # CSV로도 저장
    csv_file = base_dir / "stream_accuracy_summary.csv"
    try:
        with open(csv_file, 'w', encoding='utf-8') as f:
            # 헤더
            f.write("method,checkpoint_id,average_accuracy,memory_mb\n")
            
            # 체크포인트별 평균
            for method_name in TARGET_METHODS:
                for checkpoint_id, avg in sorted(all_results[method_name]['checkpoint_averages'].items()):
                    memory = all_results[method_name]['checkpoint_memory_averages'].get(checkpoint_id, 0)
                    f.write(f"{method_name},{checkpoint_id},{avg:.2f},{memory:.2f}\n")
            
            # 전체 평균
            for method_name in TARGET_METHODS:
                f.write(f"{method_name},overall,{all_results[method_name]['overall_average']:.2f},{all_results[method_name]['overall_memory']:.2f}\n")
        
        print(f"✅ CSV 결과 저장됨: {csv_file}")
    except Exception as e:
        print(f"❌ CSV 결과 저장 실패: {str(e)}")
    
    # 그래프 생성
    if HAS_MATPLOTLIB:
        plot_stream_accuracy_results(all_results, base_dir)
    else:
        print("\n⚠️ matplotlib이 설치되지 않아 그래프를 생성할 수 없습니다.")

def plot_stream_accuracy_results(all_results: Dict, base_dir: Path):
    """체크포인트별 정확도와 메모리 그래프 생성 (preference events 포함)"""
    # 모든 방법에서 나타나는 체크포인트 ID 수집
    all_checkpoint_ids = set()
    for method_name in TARGET_METHODS:
        all_checkpoint_ids.update(all_results[method_name]['unique_checkpoint_ids'])
    all_checkpoint_ids = sorted(all_checkpoint_ids)
    
    if not all_checkpoint_ids:
        print("⚠️ 그래프를 그릴 데이터가 없습니다.")
        return
    
    # 체크포인트별 데이터 준비
    checkpoint_data = {method_name: [] for method_name in TARGET_METHODS}
    memory_data = {method_name: [] for method_name in TARGET_METHODS}
    
    # Checkpoint ID를 docs_processed로 변환 (persona가 하나일 때)
    use_docs_processed = len(PERSONA_RANGE) == 1
    checkpoint_to_docs = {}
    if use_docs_processed:
        persona_idx = PERSONA_RANGE[0]
        for method_name in TARGET_METHODS:
            docs_by_persona = all_results[method_name].get('checkpoint_to_docs_by_persona', {})
            if persona_idx in docs_by_persona:
                checkpoint_to_docs.update(docs_by_persona[persona_idx])
                break
    
    for checkpoint_id in all_checkpoint_ids:
        for method_name in TARGET_METHODS:
            avg = all_results[method_name]['checkpoint_averages'].get(checkpoint_id, None)
            memory = all_results[method_name]['checkpoint_memory_averages'].get(checkpoint_id, None)
            checkpoint_data[method_name].append(avg if avg is not None else None)
            memory_data[method_name].append(memory if memory is not None else None)
    
    # Preference events 수집 (persona가 하나일 때만)
    all_preference_events = []
    if use_docs_processed:
        persona_idx = PERSONA_RANGE[0]
        print(f"\n  🔍 Collecting preference events for persona {persona_idx}...")
        for method_name in TARGET_METHODS:
            events_by_persona = all_results[method_name].get('preference_events_by_persona', {})
            print(f"     Method {method_name}: events_by_persona keys = {list(events_by_persona.keys())}")
            if persona_idx in events_by_persona:
                method_events = events_by_persona[persona_idx]
                print(f"     Method {method_name}: Found {len(method_events)} events")
                all_preference_events.extend(method_events)
            else:
                print(f"     Method {method_name}: No events for persona {persona_idx}")
        
        # 디버깅: preference events 확인
        if all_preference_events:
            print(f"  📌 Total preference events found: {len(all_preference_events)} events")
            for event in all_preference_events:
                print(f"     - {event.get('type', 'unknown')} at {event.get('docs_processed', 0)} docs")
        else:
            print(f"  ⚠️ No preference events found for persona {persona_idx}")
            print(f"     Available persona indices in results: {[list(all_results[m].get('preference_events_by_persona', {}).keys()) for m in TARGET_METHODS]}")
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    colors = {'standard': '#e74c3c', 'cosine': '#f39c12', 'EPIC_inst': '#2ecc71'}
    markers = {'standard': 'o', 'cosine': 's', 'EPIC_inst': '^'}
    
    # X축 결정: persona가 하나일 때는 docs_processed 사용, 아니면 checkpoint_id 사용
    if use_docs_processed and checkpoint_to_docs:
        # Checkpoint ID를 docs_processed로 변환
        x_values = [checkpoint_to_docs.get(cp_id, cp_id) for cp_id in all_checkpoint_ids]
        x_label = 'Documents Processed'
    else:
        x_values = all_checkpoint_ids
        x_label = 'Checkpoint ID'
    
    # Plot 1: Accuracy
    for method_name in TARGET_METHODS:
        accuracies = checkpoint_data[method_name]
        # None 값을 제외하고 플롯
        valid_indices = [i for i, acc in enumerate(accuracies) if acc is not None]
        valid_x = [x_values[i] for i in valid_indices]
        valid_accuracies = [accuracies[i] for i in valid_indices]
        
        if valid_accuracies:
            ax1.plot(valid_x, valid_accuracies, 
                    marker=markers[method_name], label=method_name, 
                    color=colors[method_name], linewidth=2, markersize=6)
    
    # Preference events 표시 (persona가 하나일 때만)
    add_events = []
    remove_events = []
    if use_docs_processed and all_preference_events:
        add_events = [e for e in all_preference_events if e.get('type') == 'add']
        remove_events = [e for e in all_preference_events if e.get('type') == 'remove']
        
        print(f"  📊 Plotting {len(add_events)} add events and {len(remove_events)} remove events")
        
        for event in add_events:
            docs_processed = event.get('docs_processed', 0)
            if docs_processed > 0:
                ax1.axvline(x=docs_processed, color='green', linestyle='--', 
                           alpha=0.7, linewidth=1.5, zorder=0)
        for event in remove_events:
            docs_processed = event.get('docs_processed', 0)
            if docs_processed > 0:
                ax1.axvline(x=docs_processed, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1.5, zorder=0)
    
    # Legend 설정
    handles, labels = ax1.get_legend_handles_labels()
    patches = []
    if add_events:
        patches.append(mpatches.Patch(color='green', alpha=0.7, label='Preference Added'))
    if remove_events:
        patches.append(mpatches.Patch(color='red', alpha=0.7, label='Preference Removed'))
    if patches:
        ax1.legend(handles=handles + patches, loc='best', fontsize=10)
    else:
        ax1.legend(loc='best', fontsize=10)
    
    # Title에 persona 정보 추가 (persona가 하나일 때만)
    title_suffix = f" (Persona {PERSONA_RANGE[0]})" if use_docs_processed else ""
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Stream Evaluation: Checkpoint-wise Average Accuracy{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot 2: Memory
    for method_name in TARGET_METHODS:
        memories = memory_data[method_name]
        # None 값을 제외하고 플롯
        valid_indices = [i for i, mem in enumerate(memories) if mem is not None]
        valid_x = [x_values[i] for i in valid_indices]
        valid_memories = [memories[i] for i in valid_indices]
        
        if valid_memories:
            ax2.plot(valid_x, valid_memories, 
                    marker=markers[method_name], label=method_name, 
                    color=colors[method_name], linewidth=2, markersize=6)
    
    # Preference events 표시 (persona가 하나일 때만) - ax2에도 동일하게
    if use_docs_processed and all_preference_events:
        for event in add_events:
            docs_processed = event.get('docs_processed', 0)
            if docs_processed > 0:
                ax2.axvline(x=docs_processed, color='green', linestyle='--', 
                           alpha=0.7, linewidth=1.5, zorder=0)
        for event in remove_events:
            docs_processed = event.get('docs_processed', 0)
            if docs_processed > 0:
                ax2.axvline(x=docs_processed, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1.5, zorder=0)
    
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title(f'Stream Evaluation: Checkpoint-wise Average Memory Usage{title_suffix}', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    plot_file = base_dir / "stream_accuracy_memory_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 그래프 저장됨: {plot_file}")

if __name__ == "__main__":
    main()

