#!/usr/bin/env python3
"""
KBL (Korean Benchmark for Legal Language Understanding) 데이터셋 다운로드 스크립트

save_to_disk() Arrow 형식으로 저장하여 오프라인 환경에서 사용 가능하도록 함.

Usage:
    python download_kbl.py --output-dir ./offline_datasets/kbl
    python download_kbl.py --output-dir ./offline_datasets/kbl --subsets bar_exam_civil_2024,bar_exam_criminal_2024
    python download_kbl.py --output-dir ./offline_datasets/kbl --category bar_exam
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Required packages not found. Please install:")
    print("pip install datasets tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# KBL 데이터셋 서브셋 목록 (HuggingFace lbox/kbl)
KBL_SUBSETS = {
    # Knowledge tasks (7개)
    "knowledge": [
        "kbl_knowledge_common_legal_mistake_qa",
        "kbl_knowledge_common_legal_mistake_qa_reasoning",
        "kbl_knowledge_legal_concept_qa",
        "kbl_knowledge_offense_component_qa",
        "kbl_knowledge_query_statute_matching_qa",
        "kbl_knowledge_statute_hallucination_qa",
        "kbl_knowledge_statute_number_and_content_matching_qa",
    ],
    # Reasoning tasks (4개)
    "reasoning": [
        "kbl_reasoning_case_relevance_qa_p",
        "kbl_reasoning_case_relevance_qa_q",
        "kbl_reasoning_causal_reasoning",
        "kbl_reasoning_statement_consistency_qa",
    ],
    # Bar exam - Civil (민법)
    "bar_exam_civil": [
        f"bar_exam_civil_{year}" for year in range(2012, 2026)
    ],
    # Bar exam - Criminal (형법)
    "bar_exam_criminal": [
        f"bar_exam_criminal_{year}" for year in range(2012, 2025)
    ],
    # Bar exam - Public (공법)
    "bar_exam_public": [
        f"bar_exam_public_{year}" for year in range(2012, 2025)
    ],
    # Bar exam - Responsibility (직무윤리)
    "bar_exam_responsibility": [
        f"bar_exam_responsibility_{year}" for year in range(2010, 2024)
    ],
}

# 카테고리 그룹
CATEGORY_GROUPS = {
    "all": list(KBL_SUBSETS.keys()),
    "bar_exam": ["bar_exam_civil", "bar_exam_criminal", "bar_exam_public", "bar_exam_responsibility"],
    "knowledge": ["knowledge"],
    "reasoning": ["reasoning"],
}


def get_all_subsets(categories: Optional[List[str]] = None) -> List[str]:
    """카테고리에 해당하는 모든 서브셋 반환"""
    if categories is None:
        categories = ["all"]

    subsets = []
    for category in categories:
        if category in CATEGORY_GROUPS:
            for cat_key in CATEGORY_GROUPS[category]:
                subsets.extend(KBL_SUBSETS[cat_key])
        elif category in KBL_SUBSETS:
            subsets.extend(KBL_SUBSETS[category])
        else:
            # 개별 서브셋 이름으로 처리
            subsets.append(category)

    return list(dict.fromkeys(subsets))  # 중복 제거, 순서 유지


def download_and_save(subset: str, output_dir: Path) -> dict:
    """단일 서브셋 다운로드 및 저장"""
    save_path = output_dir / subset
    result = {
        "subset": subset,
        "save_path": str(save_path),
        "success": False,
        "splits": [],
        "error": None,
    }

    try:
        logger.info(f"다운로드 중: lbox/kbl - {subset}")

        # HuggingFace에서 다운로드
        ds = load_dataset("lbox/kbl", subset, trust_remote_code=True)

        # Arrow 형식으로 저장
        ds.save_to_disk(str(save_path))

        result["success"] = True
        result["splits"] = list(ds.keys())
        result["num_examples"] = {split: len(ds[split]) for split in ds.keys()}

        logger.info(f"저장 완료: {save_path} (splits: {result['splits']})")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"실패: {subset} - {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="KBL 데이터셋을 Arrow 형식으로 다운로드 (오프라인 환경용)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="저장 디렉토리 경로"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        help="다운로드할 서브셋 (쉼표로 구분). 예: bar_exam_civil_2024,bar_exam_criminal_2024"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=list(CATEGORY_GROUPS.keys()) + list(KBL_SUBSETS.keys()),
        default="all",
        help="다운로드할 카테고리. 기본값: all"
    )
    parser.add_argument(
        "--list-subsets",
        action="store_true",
        help="사용 가능한 서브셋 목록 출력"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 서브셋 목록 출력
    if args.list_subsets:
        print("\n=== KBL 서브셋 목록 ===\n")
        for category, subsets in KBL_SUBSETS.items():
            print(f"[{category}] ({len(subsets)}개)")
            for subset in subsets:
                print(f"  - {subset}")
            print()
        print(f"총 서브셋 수: {sum(len(s) for s in KBL_SUBSETS.values())}개")
        return

    # --output-dir 필수 체크 (list-subsets가 아닐 때)
    if not args.output_dir:
        parser.error("--output-dir is required when downloading datasets")

    # 다운로드할 서브셋 결정
    if args.subsets:
        subsets = [s.strip() for s in args.subsets.split(",")]
    else:
        subsets = get_all_subsets([args.category])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"다운로드할 서브셋 수: {len(subsets)}")

    # 다운로드 실행
    results = []
    success_count = 0
    fail_count = 0

    with tqdm(total=len(subsets), desc="KBL 다운로드") as pbar:
        for subset in subsets:
            result = download_and_save(subset, output_dir)
            results.append(result)

            if result["success"]:
                success_count += 1
            else:
                fail_count += 1

            pbar.update(1)
            pbar.set_postfix({"success": success_count, "fail": fail_count})

    # 결과 요약 저장
    summary = {
        "output_dir": str(output_dir),
        "total": len(subsets),
        "success": success_count,
        "failed": fail_count,
        "results": results,
    }

    summary_path = output_dir / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 콘솔 출력
    print("\n" + "=" * 60)
    print("KBL 다운로드 완료")
    print("=" * 60)
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"저장 경로: {output_dir}")
    print(f"요약 파일: {summary_path}")

    if fail_count > 0:
        print("\n실패한 서브셋:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['subset']}: {r['error']}")

    print("\n오프라인 환경에서 사용하려면:")
    print("  1. 오프라인 YAML에서 dataset_path를 로컬 경로로 설정")
    print("  2. 환경변수 설정:")
    print("     export HF_DATASETS_OFFLINE=1")
    print("     export HF_HUB_OFFLINE=1")
    print("     export TRANSFORMERS_OFFLINE=1")


if __name__ == "__main__":
    main()
