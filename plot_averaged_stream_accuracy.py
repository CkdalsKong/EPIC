#!/usr/bin/env python3
"""
여러 stream 결과를 평균내서 정확도 그래프 그리기
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not installed. Cannot generate plots.")


def find_stream_directories(base_dir):
    """재귀적으로 모든 stream 디렉토리 찾기"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ 디렉토리가 존재하지 않습니다: {base_dir}")
        return []
    
    stream_dirs = []
    
    def search_recursive(path: Path):
        """재귀적으로 stream_ 디렉토리 찾기"""
        try:
            for item in path.iterdir():
                if item.is_dir():
                    if item.name.startswith("stream_"):
                        stream_dirs.append(str(item))
                    else:
                        search_recursive(item)
        except PermissionError:
            pass
    
    search_recursive(base_path)
    return sorted(stream_dirs)


def load_checkpoint_results(stream_dir):
    """Stream 디렉토리에서 재평가 결과 로드"""
    # 재평가 결과 우선, 없으면 원본 결과
    reevaluated_file = os.path.join(stream_dir, "all_checkpoints_reevaluated.json")
    original_file = os.path.join(stream_dir, "all_checkpoints.json")
    
    if os.path.exists(reevaluated_file):
        with open(reevaluated_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif os.path.exists(original_file):
        with open(original_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return None


def aggregate_results(all_stream_results):
    """모든 stream 결과를 checkpoint_id 기준으로 평균 계산"""
    # checkpoint_id를 키로 하는 딕셔너리
    aggregated = defaultdict(list)
    
    for stream_results in all_stream_results:
        if not stream_results:
            continue
        
        for checkpoint in stream_results:
            checkpoint_id = checkpoint.get("checkpoint_id", 0)
            docs_processed = checkpoint.get("docs_processed", 0)
            accuracy = checkpoint.get("preference_following_accuracy", 0)
            
            if checkpoint_id > 0 and accuracy is not None:
                aggregated[checkpoint_id].append({
                    "accuracy": accuracy,
                    "docs_processed": docs_processed
                })
    
    # 평균 계산 (checkpoint_id 기준)
    averaged_results = []
    for checkpoint_id in sorted(aggregated.keys()):
        data_points = aggregated[checkpoint_id]
        accuracies = [d["accuracy"] for d in data_points]
        docs_processed_list = [d["docs_processed"] for d in data_points]
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies) if len(accuracies) > 1 else 0
        avg_docs_processed = np.mean(docs_processed_list)
        
        averaged_results.append({
            "checkpoint_id": checkpoint_id,
            "docs_processed": int(round(avg_docs_processed)),
            "avg_accuracy": avg_accuracy,
            "std_accuracy": std_accuracy,
            "num_streams": len(accuracies)
        })
    
    return averaged_results


def plot_averaged_accuracy(averaged_results, output_file, pdf_only=False):
    """평균 정확도 그래프 그리기"""
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not installed. Cannot generate plots.")
        return
    
    if not averaged_results:
        print("⚠️ No data to plot.")
        return
    
    docs_processed = [r["docs_processed"] for r in averaged_results]
    avg_accuracy = [r["avg_accuracy"] for r in averaged_results]
    std_accuracy = [r["std_accuracy"] for r in averaged_results]
    
    # 완전 처음 크기에서 조금만 키움
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 평균 선 그래프
    ax.plot(docs_processed, avg_accuracy, marker='o', linewidth=2.5, 
            markersize=10, label='Average Accuracy', color='#2ecc71', zorder=3)
    
    # 표준편차 영역 (신뢰구간)
    if any(std_accuracy):
        upper_bound = [a + s for a, s in zip(avg_accuracy, std_accuracy)]
        lower_bound = [a - s for a, s in zip(avg_accuracy, std_accuracy)]
        ax.fill_between(docs_processed, lower_bound, upper_bound,
                       alpha=0.25, color='#2ecc71', label='±1 Std Dev', zorder=1)
    
    ax.set_xlabel('Documents Processed', fontsize=20, fontweight='bold')
    ax.set_ylabel('Preference Following Accuracy (%)', fontsize=20, fontweight='bold')
    # ax.set_title('Averaged Stream Evaluation: Preference Following Accuracy', 
    #             fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=16, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 100)
    
    # X축 눈금을 더 깔끔하게
    ax.set_xticks(docs_processed)
    ax.tick_params(axis='both', labelsize=16)
    
    # 각 점에 개수 표시 (선택적, 너무 많으면 생략)
    if len(averaged_results) <= 10:
        for i, r in enumerate(averaged_results):
            ax.annotate(f'n={r["num_streams"]}', 
                       (r["docs_processed"], r["avg_accuracy"]),
                       textcoords="offset points", 
                       xytext=(0,12), 
                       ha='center', fontsize=14, alpha=0.6,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # PDF로 저장
    if output_file.endswith('.png'):
        pdf_file = output_file.replace('.png', '.pdf')
    elif output_file.endswith('.pdf'):
        pdf_file = output_file
    else:
        pdf_file = output_file + '.pdf'
    
    plt.savefig(pdf_file, dpi=150, bbox_inches='tight', format='pdf')
    print(f"📈 평균 정확도 그래프 저장 (PDF): {pdf_file}")
    
    # PNG도 저장 (pdf_only가 False일 때만)
    if not pdf_only and not output_file.endswith('.pdf'):
        png_file = output_file if output_file.endswith('.png') else output_file + '.png'
        plt.savefig(png_file, dpi=150, bbox_inches='tight')
        print(f"📈 평균 정확도 그래프 저장 (PNG): {png_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='여러 stream 결과를 평균내서 정확도 그래프 그리기')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='기본 디렉토리 (예: stream_prefeval/lmsys_sampled/EPIC_inst/)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='출력 파일 경로 (기본값: base_dir/averaged_accuracy_plot.png, PDF도 자동 생성)')
    parser.add_argument('--pdf_only', action='store_true',
                       help='PDF만 저장하고 PNG는 저장하지 않음')
    
    args = parser.parse_args()
    
    # Stream 디렉토리 찾기
    stream_dirs = find_stream_directories(args.base_dir)
    
    if not stream_dirs:
        print(f"❌ Stream 디렉토리를 찾을 수 없습니다: {args.base_dir}")
        return
    
    print(f"✅ {len(stream_dirs)}개의 Stream 디렉토리 발견")
    
    # 각 stream 결과 로드
    all_stream_results = []
    for stream_dir in stream_dirs:
        results = load_checkpoint_results(stream_dir)
        if results:
            all_stream_results.append(results)
            print(f"   ✓ {os.path.basename(stream_dir)}: {len(results)} checkpoints")
        else:
            print(f"   ⚠️ {os.path.basename(stream_dir)}: 결과 파일 없음")
    
    if not all_stream_results:
        print("❌ 로드할 수 있는 결과가 없습니다")
        return
    
    # 평균 계산
    averaged_results = aggregate_results(all_stream_results)
    
    if not averaged_results:
        print("❌ 평균 계산할 데이터가 없습니다")
        return
    
    print(f"\n📊 평균 결과:")
    for r in averaged_results:
        print(f"   {r['docs_processed']} docs: {r['avg_accuracy']:.2f}% ± {r['std_accuracy']:.2f}% (n={r['num_streams']})")
    
    # 그래프 그리기
    if args.output_file:
        output_file = args.output_file
    else:
        base_path = Path(args.base_dir)
        output_file = str(base_path / "averaged_accuracy_plot.png")
    
    plot_averaged_accuracy(averaged_results, output_file, pdf_only=args.pdf_only)
    
    # 결과 저장
    results_file = str(Path(output_file).parent / "averaged_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(averaged_results, f, ensure_ascii=False, indent=2)
    print(f"📄 평균 결과 저장: {results_file}")
    
    print("\n✅ 완료")


if __name__ == "__main__":
    main()

