#!/usr/bin/env python3
"""
Stream 결과를 다른 LLM으로 재평가하는 스크립트
EPIC_stream.py로 생성한 결과를 evaluation_with_different_llm.py로 평가
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not installed. Plotting functions will be disabled.")

from evaluation_with_different_llm import EvaluationWithDifferentLLM


class StreamEvaluator:
    """Stream 결과를 다른 LLM으로 재평가하는 클래스"""
    
    def __init__(self, evaluator: EvaluationWithDifferentLLM):
        """
        Args:
            evaluator: EvaluationWithDifferentLLM 인스턴스
        """
        self.evaluator = evaluator
    
    def find_stream_directories(self, base_dir: str) -> List[str]:
        """
        Stream 디렉토리들을 찾기
        
        Args:
            base_dir: 메소드 디렉토리 (예: output_prefeval/EPIC_inst/1)
        
        Returns:
            List of stream directory paths
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"❌ 디렉토리가 존재하지 않습니다: {base_dir}")
            return []
        
        # stream_으로 시작하는 디렉토리 찾기
        stream_dirs = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("stream_"):
                stream_dirs.append(str(item))
        
        # 최신 순으로 정렬
        stream_dirs.sort(reverse=True)
        
        if not stream_dirs:
            print(f"⚠️ Stream 디렉토리를 찾을 수 없습니다: {base_dir}")
        else:
            print(f"✅ {len(stream_dirs)}개의 Stream 디렉토리 발견")
            for sd in stream_dirs:
                print(f"   - {os.path.basename(sd)}")
        
        return stream_dirs
    
    def find_checkpoints(self, stream_dir: str) -> List[Dict[str, Any]]:
        """
        Stream 디렉토리에서 체크포인트들 찾기
        
        Args:
            stream_dir: Stream 디렉토리 경로
        
        Returns:
            List of checkpoint info dicts: [{"id": int, "dir": str, "generation_file": str}]
        """
        stream_path = Path(stream_dir)
        checkpoints = []
        
        # checkpoint_으로 시작하는 디렉토리 찾기
        for item in stream_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_"):
                checkpoint_id = int(item.name.split("_")[1])
                generation_file = item / "generation_results.json"
                
                if generation_file.exists():
                    checkpoints.append({
                        "id": checkpoint_id,
                        "dir": str(item),
                        "generation_file": str(generation_file)
                    })
        
        # 체크포인트 ID 순으로 정렬
        checkpoints.sort(key=lambda x: x["id"])
        
        print(f"✅ {len(checkpoints)}개의 체크포인트 발견")
        for cp in checkpoints:
            print(f"   - Checkpoint {cp['id']}: {os.path.basename(cp['dir'])}")
        
        return checkpoints
    
    def evaluate_checkpoint(self, checkpoint_info: Dict[str, Any], 
                           overwrite: bool = False) -> Optional[Dict[str, Any]]:
        """
        단일 체크포인트 평가
        
        Args:
            checkpoint_info: 체크포인트 정보
            overwrite: 기존 평가 결과를 덮어쓸지 여부
        
        Returns:
            평가 결과 메트릭 또는 None
        """
        checkpoint_id = checkpoint_info["id"]
        checkpoint_dir = checkpoint_info["dir"]
        generation_file = checkpoint_info["generation_file"]
        
        # 출력 파일 경로
        output_file = os.path.join(checkpoint_dir, "generation_results_evaluated.json")
        
        # 이미 평가된 경우
        if os.path.exists(output_file) and not overwrite:
            print(f"   ⏭️ Checkpoint {checkpoint_id} 이미 평가됨 (건너뛰기)")
            try:
                stats = self.evaluator.analyze_evaluation_results(output_file)
                return {
                    "checkpoint_id": checkpoint_id,
                    "stats": stats,
                    "evaluated_file": output_file
                }
            except Exception as e:
                print(f"   ⚠️ 기존 결과 로드 실패, 재평가: {str(e)}")
        
        print(f"   🔄 Checkpoint {checkpoint_id} 평가 중...")
        
        # 평가 수행
        evaluated_file = self.evaluator.evaluate_generation_file(
            generation_file, 
            output_file
        )
        
        if not evaluated_file:
            print(f"   ❌ Checkpoint {checkpoint_id} 평가 실패")
            return None
        
        # 결과 분석
        stats = self.evaluator.analyze_evaluation_results(evaluated_file)
        
        # 메트릭 계산 (EPIC_stream.py 형식에 맞춤)
        total = stats.get("total_responses", 0)
        if total == 0:
            print(f"   ⚠️ Checkpoint {checkpoint_id}: 평가 결과가 없습니다")
            return None
        
        # Stream 형식의 메트릭 생성
        metrics = {
            "checkpoint_id": checkpoint_id,
            "unhelpful": stats.get("error_unhelpful", 0),
            "inconsistent": stats.get("error_inconsistent", 0),
            "hallucination_of_preference_violation": stats.get("hallucination_of_preference_violation", 0),
            "preference_unaware_violation": stats.get("preference_unaware_violation", 0),
            "preference_following_accuracy": stats.get("preference_following_accuracy_percent", 0),
            "total_responses": total,
            "evaluated_file": evaluated_file
        }
        
        print(f"   ✅ Checkpoint {checkpoint_id} 평가 완료 - 정확도: {metrics['preference_following_accuracy']:.2f}%")
        
        return metrics
    
    def load_stream_metadata(self, stream_dir: str) -> Dict[str, Any]:
        """Stream 메타데이터 로드"""
        meta_file = os.path.join(stream_dir, "stream_metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 메타데이터 로드 실패: {str(e)}")
        return {}
    
    def load_preference_history(self, stream_dir: str) -> List[Dict[str, Any]]:
        """Preference 히스토리 로드"""
        history_file = os.path.join(stream_dir, "preference_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Preference 히스토리 로드 실패: {str(e)}")
        return []
    
    def load_checkpoint_metadata(self, checkpoint_dir: str) -> Dict[str, Any]:
        """체크포인트 메타데이터 로드 (docs_processed 등)"""
        metrics_file = os.path.join(checkpoint_dir, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 체크포인트 메타데이터 로드 실패: {str(e)}")
        return {}
    
    def evaluate_stream(self, stream_dir: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Stream 디렉토리의 모든 체크포인트 평가
        
        Args:
            stream_dir: Stream 디렉토리 경로
            overwrite: 기존 평가 결과를 덮어쓸지 여부
        
        Returns:
            평가 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"🚀 Stream 재평가 시작: {os.path.basename(stream_dir)}")
        print(f"{'='*60}")
        
        # 체크포인트 찾기
        checkpoints = self.find_checkpoints(stream_dir)
        if not checkpoints:
            print("❌ 평가할 체크포인트가 없습니다")
            return {}
        
        # Stream 메타데이터 로드
        stream_meta = self.load_stream_metadata(stream_dir)
        preference_history = self.load_preference_history(stream_dir)
        
        # 각 체크포인트 평가
        checkpoint_results = []
        for checkpoint_info in tqdm(checkpoints, desc="체크포인트 평가"):
            # 체크포인트 메타데이터 로드 (docs_processed 등)
            checkpoint_meta = self.load_checkpoint_metadata(checkpoint_info["dir"])
            
            # 평가 수행
            metrics = self.evaluate_checkpoint(checkpoint_info, overwrite)
            
            if metrics:
                # 메타데이터 추가
                metrics["docs_processed"] = checkpoint_meta.get("docs_processed", 0)
                metrics["total_indexed"] = checkpoint_meta.get("total_indexed", 0)
                metrics["active_chunks"] = checkpoint_meta.get("active_chunks", 0)
                metrics["active_preferences"] = checkpoint_meta.get("active_preferences", 0)
                metrics["timestamp"] = datetime.now().isoformat()
                
                checkpoint_results.append(metrics)
        
        # 결과 저장
        results_file = os.path.join(stream_dir, "all_checkpoints_reevaluated.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 재평가 결과 저장: {results_file}")
        
        # CSV 생성
        self._generate_summary_csv(stream_dir, checkpoint_results)
        
        # 그래프 생성
        if HAS_MATPLOTLIB:
            self.plot_stream_results(stream_dir, checkpoint_results, preference_history)
        
        return {
            "stream_dir": stream_dir,
            "checkpoints": checkpoint_results,
            "metadata": stream_meta
        }
    
    def _generate_summary_csv(self, stream_dir: str, checkpoint_results: List[Dict[str, Any]]):
        """CSV 요약 생성"""
        csv_file = os.path.join(stream_dir, "checkpoint_summary_reevaluated.csv")
        
        fieldnames = [
            "checkpoint_id",
            "docs_processed",
            "total_indexed",
            "active_chunks",
            "active_preferences",
            "unhelpful",
            "inconsistent",
            "hallucination_of_preference_violation",
            "preference_unaware_violation",
            "preference_following_accuracy",
            "total_responses",
            "timestamp"
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for checkpoint in checkpoint_results:
                row = {field: checkpoint.get(field, "") for field in fieldnames}
                writer.writerow(row)
        
        print(f"📊 재평가 CSV 저장: {csv_file}")
    
    def plot_stream_results(self, stream_dir: str, checkpoint_results: List[Dict[str, Any]], 
                           preference_history: List[Dict[str, Any]]):
        """
        Stream 결과 그래프 생성 (EPIC_stream.py와 동일한 형식)
        """
        if not HAS_MATPLOTLIB:
            print("⚠️ matplotlib not installed. Cannot generate plots.")
            return
        
        if not checkpoint_results:
            print("⚠️ No checkpoint results to plot.")
            return
        
        # 데이터 추출
        docs_processed = [cp.get("docs_processed", 0) for cp in checkpoint_results]
        
        metrics = {
            "Unhelpful": [cp.get("unhelpful", 0) for cp in checkpoint_results],
            "Inconsistent": [cp.get("inconsistent", 0) for cp in checkpoint_results],
            "Hallucination Violation": [cp.get("hallucination_of_preference_violation", 0) for cp in checkpoint_results],
            "Unaware Violation": [cp.get("preference_unaware_violation", 0) for cp in checkpoint_results],
            "Accuracy (%)": [cp.get("preference_following_accuracy", 0) for cp in checkpoint_results]
        }
        
        # 그래프 1: 2개 subplot (Error metrics + Accuracy)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71']
        
        # Plot 1: Error metrics (counts)
        ax1 = axes[0]
        for i, (metric_name, values) in enumerate(list(metrics.items())[:-1]):
            ax1.plot(docs_processed, values, marker='o', label=metric_name,
                    color=colors[i], linewidth=2, markersize=6)
        
        ax1.set_ylabel('Error Count', fontsize=12)
        ax1.set_title('Stream Evaluation (Re-evaluated): Error Metrics Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2 = axes[1]
        ax2.plot(docs_processed, metrics["Accuracy (%)"], marker='s',
                color=colors[4], linewidth=2, markersize=8, label='Preference Following Accuracy')
        ax2.fill_between(docs_processed, metrics["Accuracy (%)"], alpha=0.3, color=colors[4])
        
        ax2.set_xlabel('Documents Processed', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Stream Evaluation (Re-evaluated): Preference Following Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Preference events 추가
        add_events = [e for e in preference_history if e.get("type") == "add"]
        remove_events = [e for e in preference_history if e.get("type") == "remove"]
        
        for ax in axes:
            for event in add_events:
                docs_at_event = event.get("docs_processed", 0)
                if docs_at_event > 0:
                    ax.axvline(x=docs_at_event, color='green', linestyle='--',
                              alpha=0.7, linewidth=1.5)
            for event in remove_events:
                docs_at_event = event.get("docs_processed", 0)
                if docs_at_event > 0:
                    ax.axvline(x=docs_at_event, color='red', linestyle='--',
                              alpha=0.7, linewidth=1.5)
        
        # Legend for events
        if add_events or remove_events:
            add_patch = mpatches.Patch(color='green', alpha=0.7, label='Preference Added')
            remove_patch = mpatches.Patch(color='red', alpha=0.7, label='Preference Removed')
            patches = []
            if add_events:
                patches.append(add_patch)
            if remove_events:
                patches.append(remove_patch)
            if patches:
                axes[0].legend(handles=list(axes[0].get_legend_handles_labels()[0]) + patches,
                              loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(stream_dir, "stream_evaluation_reevaluated_plot.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 그래프 저장: {plot_file}")
        
        # Combined metrics plot
        self._plot_combined_metrics(stream_dir, docs_processed, metrics, preference_history)
    
    def _plot_combined_metrics(self, stream_dir: str, docs_processed: List[int],
                              metrics: Dict[str, List[float]], preference_history: List[Dict[str, Any]]):
        """Combined metrics plot 생성"""
        if not HAS_MATPLOTLIB:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71']
        markers = ['o', 's', '^', 'D', 'v']
        
        # Normalize error counts to percentage
        max_errors = max(
            max(metrics["Unhelpful"]) if metrics["Unhelpful"] else 1,
            max(metrics["Inconsistent"]) if metrics["Inconsistent"] else 1,
            max(metrics["Hallucination Violation"]) if metrics["Hallucination Violation"] else 1,
            max(metrics["Unaware Violation"]) if metrics["Unaware Violation"] else 1,
            1
        )
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            if metric_name == "Accuracy (%)":
                plot_values = values
            else:
                # Invert and normalize: fewer errors = higher score
                plot_values = [100 - (v / max_errors * 100) if max_errors > 0 else 100 for v in values]
                metric_name = f"No {metric_name} (%)"
            
            ax.plot(docs_processed, plot_values, marker=markers[i], label=metric_name,
                   color=colors[i], linewidth=2, markersize=6)
        
        # Add preference events
        for event in preference_history:
            docs_at_event = event.get("docs_processed", 0)
            if docs_at_event > 0:
                color = 'green' if event.get("type") == "add" else 'red'
                ax.axvline(x=docs_at_event, color=color, linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Documents Processed', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Stream Evaluation (Re-evaluated): All Metrics (Higher is Better)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        plot_file = os.path.join(stream_dir, "stream_combined_reevaluated_plot.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Combined 그래프 저장: {plot_file}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Stream 결과를 다른 LLM으로 재평가')
    parser.add_argument('--stream_dir', type=str, default=None,
                       help='Stream 디렉토리 경로 (지정하지 않으면 base_dir에서 자동 검색)')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='메소드 디렉토리 (예: output_prefeval/EPIC_inst/1)')
    parser.add_argument('--vllm_url', type=str, default='http://localhost:8011',
                       help='vLLM 서버 URL')
    parser.add_argument('--eval_model', type=str, default='meta-llama/Llama-3.3-70B-Instruct',
                       help='평가 모델')
    parser.add_argument('--overwrite', action='store_true',
                       help='기존 평가 결과를 덮어쓰기')
    
    args = parser.parse_args()
    
    # 평가기 초기화
    evaluator = EvaluationWithDifferentLLM(
        vllm_base_url=args.vllm_url,
        evaluation_model=args.eval_model,
        max_tokens=512,
        temperature=0.0,
        timeout=60,
        retry_count=3
    )
    
    stream_evaluator = StreamEvaluator(evaluator)
    
    # Stream 디렉토리 찾기
    if args.stream_dir:
        stream_dirs = [args.stream_dir]
    elif args.base_dir:
        stream_dirs = stream_evaluator.find_stream_directories(args.base_dir)
    else:
        print("❌ --stream_dir 또는 --base_dir 중 하나를 지정해야 합니다")
        return
    
    if not stream_dirs:
        print("❌ 평가할 Stream 디렉토리가 없습니다")
        return
    
    # 각 Stream 디렉토리 평가
    for stream_dir in stream_dirs:
        try:
            stream_evaluator.evaluate_stream(stream_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"❌ Stream 평가 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ 모든 Stream 재평가 완료")


if __name__ == "__main__":
    main()

