#!/usr/bin/env python3
"""
이미 처리된 Stream 결과를 새로운 preference events로 재처리하는 스크립트
"""

import os
import json
import argparse
from pathlib import Path
from EPIC_stream import StreamSetup, StreamManager
from EPIC_utils import EPICUtils


def load_stream_setup(stream_dir, utils):
    """기존 Stream 디렉토리에서 StreamSetup 복원"""
    stream = StreamSetup(utils, batch_size=2000)
    
    # Find latest checkpoint to restore from
    checkpoint_dirs = [d for d in os.listdir(stream_dir) 
                      if d.startswith("checkpoint_") and os.path.isdir(os.path.join(stream_dir, d))]
    if not checkpoint_dirs:
        print("❌ No checkpoints found")
        return None
    
    checkpoint_dirs.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
    
    # Load metadata from first checkpoint to get persona_index
    first_checkpoint = os.path.join(stream_dir, checkpoint_dirs[0])
    pref_state_file = os.path.join(first_checkpoint, "preference_state.json")
    
    if not os.path.exists(pref_state_file):
        print("❌ Cannot find preference state file")
        return None
    
    # We need to restore from a checkpoint
    # For now, we'll need the original chunks and embeddings
    # This is a simplified version - full restoration would need original data
    
    return stream


def reprocess_stream_preferences(stream_dir, new_preference_events, utils, 
                                start_from_checkpoint=None):
    """
    Stream 결과를 새로운 preference events로 재처리
    
    Args:
        stream_dir: Stream 디렉토리 경로
        new_preference_events: 새로운 preference events 리스트
        utils: EPICUtils 인스턴스
        start_from_checkpoint: 시작할 체크포인트 ID (None = 처음부터)
    """
    print(f"\n{'='*60}")
    print(f"🔄 Stream Preference 재처리")
    print(f"{'='*60}")
    print(f"Stream 디렉토리: {stream_dir}")
    print(f"새로운 preference events: {len(new_preference_events)}개")
    
    # Load stream metadata
    meta_file = os.path.join(stream_dir, "stream_metadata.json")
    if not os.path.exists(meta_file):
        print(f"❌ Stream metadata를 찾을 수 없습니다: {meta_file}")
        return
    
    stream_meta = utils.load_json(meta_file)
    original_events = stream_meta.get("preference_events", [])
    
    print(f"\n📊 원본 preference events: {len(original_events)}개")
    for event in original_events:
        print(f"   - {event.get('type', 'unknown')} at {event.get('docs_processed', 0)} docs")
    
    print(f"\n📊 새로운 preference events: {len(new_preference_events)}개")
    for event in new_preference_events:
        print(f"   - {event.get('type', 'unknown')} at {event.get('at_docs', 0)} docs")
    
    # Find checkpoints
    checkpoint_dirs = [d for d in os.listdir(stream_dir) 
                      if d.startswith("checkpoint_") and os.path.isdir(os.path.join(stream_dir, d))]
    checkpoint_dirs.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
    
    if not checkpoint_dirs:
        print("❌ 체크포인트를 찾을 수 없습니다")
        return
    
    print(f"\n✅ {len(checkpoint_dirs)}개의 체크포인트 발견")
    
    # Save new preference events
    new_events_file = os.path.join(stream_dir, "preference_events_new.json")
    utils.save_json(new_events_file, new_preference_events)
    print(f"💾 새로운 preference events 저장: {new_events_file}")
    
    print(f"\n⚠️  주의: 완전한 재처리를 위해서는 원본 데이터(chunks, embeddings)가 필요합니다.")
    print(f"   현재는 preference events만 업데이트합니다.")
    print(f"   체크포인트 재평가는 수동으로 수행해야 합니다.")
    
    # Update stream metadata with new events (for reference)
    stream_meta["preference_events"] = new_preference_events
    stream_meta["preference_events_original"] = original_events
    utils.save_json(meta_file, stream_meta)
    
    print(f"\n✅ Preference events 업데이트 완료")
    print(f"   체크포인트 재평가는 evaluate_stream_with_different_llm.py를 사용하세요")


def main():
    parser = argparse.ArgumentParser(description='Stream preference events 재처리')
    parser.add_argument('--stream_dir', type=str, required=True,
                       help='Stream 디렉토리 경로')
    parser.add_argument('--num_add', type=int, default=1,
                       help='추가할 preference 개수')
    parser.add_argument('--num_remove', type=int, default=1,
                       help='제거할 preference 개수')
    parser.add_argument('--total_docs', type=int, default=10000,
                       help='전체 문서 수')
    parser.add_argument('--batch_size', type=int, default=2000,
                       help='체크포인트 간격 (batch size)')
    parser.add_argument('--seed', type=int, default=None,
                       help='랜덤 시드 (재현성)')
    parser.add_argument('--preference_events_file', type=str, default=None,
                       help='Preference events JSON 파일 (직접 지정)')
    
    args = parser.parse_args()
    
    # Load or create preference events
    if args.preference_events_file and os.path.exists(args.preference_events_file):
        with open(args.preference_events_file, 'r', encoding='utf-8') as f:
            new_preference_events = json.load(f)
        print(f"✅ Preference events 파일 로드: {args.preference_events_file}")
    else:
        # Create new random preference events
        from EPIC_stream import StreamManager
        
        # Create a dummy utils (we only need it for the manager)
        # In practice, you'd want to load the actual utils used
        class DummyUtils:
            pass
        
        dummy_utils = DummyUtils()
        manager = StreamManager(dummy_utils)
        
        new_preference_events = manager.create_fixed_preference_events(
            batch_size=args.batch_size,
            total_docs=args.total_docs,
            num_add=args.num_add,
            num_remove=args.num_remove,
            seed=args.seed
        )
    
    # For reprocessing, we need the actual utils
    # This is a simplified version - you may need to adjust based on your setup
    print(f"\n⚠️  완전한 재처리를 위해서는 원본 EPICUtils 인스턴스가 필요합니다.")
    print(f"   현재는 preference events만 업데이트합니다.")
    
    # Create minimal utils for file operations
    class MinimalUtils:
        def load_json(self, path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        def save_json(self, path, data):
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    utils = MinimalUtils()
    
    reprocess_stream_preferences(
        args.stream_dir,
        new_preference_events,
        utils
    )


if __name__ == "__main__":
    main()

