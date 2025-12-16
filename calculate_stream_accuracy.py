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

def find_stream_dirs(base_dir: Path, method: str, persona_index: int) -> List[Path]:
    """특정 방법과 persona_index에 대한 stream 디렉토리들을 찾기"""
    # base_dir 구조 확인
    # 경우 1: base_dir이 이미 method 디렉토리 (예: stream_prefwiki/wiki/EPIC_inst/)
    if base_dir.name == method:
        method_dir = base_dir / str(persona_index)
    # 경우 2: base_dir이 method 디렉토리의 부모 (예: stream_prefwiki/wiki/)
    elif (base_dir / method).exists():
        method_dir = base_dir / method / str(persona_index)
    # 경우 3: base_dir이 루트 디렉토리 (예: . 또는 workspace root)
    else:
        method_dir = base_dir / "stream_prefwiki" / "wiki" / method / str(persona_index)
    
    if not method_dir.exists():
        return []
    
    # stream_* 디렉토리들 찾기
    stream_dirs = [d for d in method_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('stream_')]
    
    return stream_dirs

def load_reevaluated_results(stream_dir: Path) -> List[Dict]:
    """재평가 결과 JSON 파일 로드"""
    results_file = stream_dir / "all_checkpoints_reevaluated.json"
    
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

def calculate_checkpoint_accuracy(
    base_dir: Path,
    method: str
) -> Tuple[Dict[int, List[float]], List[float], Dict[int, List[float]]]:
    """
    특정 방법에 대한 체크포인트별 정확도 및 메모리 수집
    
    Returns:
        (checkpoint_accuracies, all_accuracies, checkpoint_memories)
        - checkpoint_accuracies: {checkpoint_id: [accuracy1, accuracy2, ...]}
        - all_accuracies: 모든 체크포인트의 모든 정확도 리스트
        - checkpoint_memories: {checkpoint_id: [memory1, memory2, ...]}
    """
    checkpoint_accuracies = defaultdict(list)
    all_accuracies = []
    checkpoint_memories = defaultdict(list)
    
    print(f"\n{'='*80}")
    print(f"📊 {method} 방법 분석 중...")
    print(f"{'='*80}")
    
    valid_personas = 0
    
    for persona_index in PERSONA_RANGE:
        stream_dirs = find_stream_dirs(base_dir, method, persona_index)
        
        if not stream_dirs:
            continue
        
        # 가장 최근 stream 디렉토리 사용 (또는 모든 디렉토리)
        # 여기서는 가장 최근 것만 사용
        stream_dir = max(stream_dirs, key=lambda x: x.stat().st_mtime)
        
        results = load_reevaluated_results(stream_dir)
        
        if not results:
            continue
        
        valid_personas += 1
        persona_checkpoints = {}
        
        for checkpoint in results:
            checkpoint_id = checkpoint.get('checkpoint_id')
            accuracy = checkpoint.get('preference_following_accuracy')
            memory_mb = checkpoint.get('memory_mb', 0)
            
            if checkpoint_id is not None and accuracy is not None:
                checkpoint_accuracies[checkpoint_id].append(accuracy)
                all_accuracies.append(accuracy)
                persona_checkpoints[checkpoint_id] = accuracy
                
                # Collect memory information
                if memory_mb > 0:
                    checkpoint_memories[checkpoint_id].append(memory_mb)
        
        if persona_index % 10 == 0:
            print(f"  Persona {persona_index}: {len(results)} checkpoints found")
    
    print(f"  ✅ 유효한 Persona 수: {valid_personas}/{len(PERSONA_RANGE)}")
    
    return checkpoint_accuracies, all_accuracies, checkpoint_memories

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
    for method in TARGET_METHODS:
        checkpoint_accuracies, all_accuracies, checkpoint_memories = calculate_checkpoint_accuracy(
            base_dir, method
        )
        
        # 체크포인트별 평균 계산
        checkpoint_averages = calculate_averages(checkpoint_accuracies)
        checkpoint_memory_averages = calculate_averages(checkpoint_memories)
        
        # 전체 평균 계산
        overall_avg = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        overall_memory = sum([sum(memories) for memories in checkpoint_memories.values()]) / sum([len(memories) for memories in checkpoint_memories.values()]) if checkpoint_memories else 0.0
        
        all_results[method] = {
            'checkpoint_averages': checkpoint_averages,
            'checkpoint_memory_averages': checkpoint_memory_averages,
            'overall_average': overall_avg,
            'overall_memory': overall_memory,
            'total_checkpoints': len(all_accuracies),
            'unique_checkpoint_ids': sorted(checkpoint_averages.keys())
        }
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"📋 체크포인트별 평균 정확도")
    print(f"{'='*80}")
    
    # 모든 방법에서 나타나는 체크포인트 ID 수집
    all_checkpoint_ids = set()
    for method in TARGET_METHODS:
        all_checkpoint_ids.update(all_results[method]['unique_checkpoint_ids'])
    all_checkpoint_ids = sorted(all_checkpoint_ids)
    
    # 헤더
    header = f"{'Checkpoint':<12} "
    for method in TARGET_METHODS:
        header += f"{method:>15} "
    print(header)
    print("-" * 60)
    
    # 체크포인트별 결과
    for checkpoint_id in all_checkpoint_ids:
        row = f"{checkpoint_id:<12} "
        for method in TARGET_METHODS:
            avg = all_results[method]['checkpoint_averages'].get(checkpoint_id, None)
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
    
    for method in TARGET_METHODS:
        overall_avg = all_results[method]['overall_average']
        overall_memory = all_results[method]['overall_memory']
        total_checkpoints = all_results[method]['total_checkpoints']
        print(f"{method:15s}: {overall_avg:6.2f}% | 메모리: {overall_memory:6.2f} MB (총 {total_checkpoints} 체크포인트)")
    
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
            for method in TARGET_METHODS:
                for checkpoint_id, avg in sorted(all_results[method]['checkpoint_averages'].items()):
                    memory = all_results[method]['checkpoint_memory_averages'].get(checkpoint_id, 0)
                    f.write(f"{method},{checkpoint_id},{avg:.2f},{memory:.2f}\n")
            
            # 전체 평균
            for method in TARGET_METHODS:
                f.write(f"{method},overall,{all_results[method]['overall_average']:.2f},{all_results[method]['overall_memory']:.2f}\n")
        
        print(f"✅ CSV 결과 저장됨: {csv_file}")
    except Exception as e:
        print(f"❌ CSV 결과 저장 실패: {str(e)}")
    
    # 그래프 생성
    if HAS_MATPLOTLIB:
        plot_stream_accuracy_results(all_results, base_dir)
    else:
        print("\n⚠️ matplotlib이 설치되지 않아 그래프를 생성할 수 없습니다.")

def plot_stream_accuracy_results(all_results: Dict, base_dir: Path):
    """체크포인트별 정확도와 메모리 그래프 생성"""
    # 모든 방법에서 나타나는 체크포인트 ID 수집
    all_checkpoint_ids = set()
    for method in TARGET_METHODS:
        all_checkpoint_ids.update(all_results[method]['unique_checkpoint_ids'])
    all_checkpoint_ids = sorted(all_checkpoint_ids)
    
    if not all_checkpoint_ids:
        print("⚠️ 그래프를 그릴 데이터가 없습니다.")
        return
    
    # 체크포인트별 데이터 준비
    checkpoint_data = {method: [] for method in TARGET_METHODS}
    memory_data = {method: [] for method in TARGET_METHODS}
    
    for checkpoint_id in all_checkpoint_ids:
        for method in TARGET_METHODS:
            avg = all_results[method]['checkpoint_averages'].get(checkpoint_id, None)
            memory = all_results[method]['checkpoint_memory_averages'].get(checkpoint_id, None)
            checkpoint_data[method].append(avg if avg is not None else None)
            memory_data[method].append(memory if memory is not None else None)
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    colors = {'standard': '#e74c3c', 'cosine': '#f39c12', 'EPIC_inst': '#2ecc71'}
    markers = {'standard': 'o', 'cosine': 's', 'EPIC_inst': '^'}
    
    # Plot 1: Accuracy
    for method in TARGET_METHODS:
        accuracies = checkpoint_data[method]
        # None 값을 제외하고 플롯
        valid_indices = [i for i, acc in enumerate(accuracies) if acc is not None]
        valid_checkpoints = [all_checkpoint_ids[i] for i in valid_indices]
        valid_accuracies = [accuracies[i] for i in valid_indices]
        
        if valid_accuracies:
            ax1.plot(valid_checkpoints, valid_accuracies, 
                    marker=markers[method], label=method, 
                    color=colors[method], linewidth=2, markersize=6)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Stream Evaluation: Checkpoint-wise Average Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot 2: Memory
    for method in TARGET_METHODS:
        memories = memory_data[method]
        # None 값을 제외하고 플롯
        valid_indices = [i for i, mem in enumerate(memories) if mem is not None]
        valid_checkpoints = [all_checkpoint_ids[i] for i in valid_indices]
        valid_memories = [memories[i] for i in valid_indices]
        
        if valid_memories:
            ax2.plot(valid_checkpoints, valid_memories, 
                    marker=markers[method], label=method, 
                    color=colors[method], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Checkpoint ID', fontsize=12)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Stream Evaluation: Checkpoint-wise Average Memory Usage', fontsize=14, fontweight='bold')
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

