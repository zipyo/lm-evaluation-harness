#!/usr/bin/env python3
"""
다운로드된 데이터셋 무결성 검증 스크립트

Usage:
    python verify_datasets.py --data-dir ./offline_datasets
"""

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from datasets import load_from_disk
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


def verify_dataset(dataset_path: Path) -> dict:
    """단일 데이터셋 무결성 검증"""
    result = {
        "path": str(dataset_path),
        "valid": False,
        "splits": [],
        "num_examples": {},
        "error": None,
    }

    try:
        ds = load_from_disk(str(dataset_path))
        result["valid"] = True
        result["splits"] = list(ds.keys())
        result["num_examples"] = {split: len(ds[split]) for split in ds.keys()}
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="다운로드된 데이터셋 무결성 검증"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="데이터셋 디렉토리 경로"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "mmlu", "kmmlu", "kmmlu_hard", "kbl"],
        default="all",
        help="검증할 벤치마크"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        sys.exit(1)

    # 검증할 벤치마크 디렉토리 목록
    if args.benchmark == "all":
        benchmark_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
    else:
        benchmark_dirs = [data_dir / args.benchmark]

    total_valid = 0
    total_invalid = 0
    all_results = {}

    for bench_dir in benchmark_dirs:
        if not bench_dir.exists():
            logger.warning(f"벤치마크 디렉토리 없음: {bench_dir}")
            continue

        bench_name = bench_dir.name
        logger.info(f"=== {bench_name.upper()} 검증 ===")

        dataset_dirs = [d for d in bench_dir.iterdir() if d.is_dir()]
        results = []

        with tqdm(total=len(dataset_dirs), desc=f"{bench_name} 검증") as pbar:
            for ds_dir in dataset_dirs:
                result = verify_dataset(ds_dir)
                results.append(result)

                if result["valid"]:
                    total_valid += 1
                else:
                    total_invalid += 1
                    logger.error(f"✗ 손상됨: {ds_dir.name} - {result['error']}")

                pbar.update(1)
                pbar.set_postfix({"valid": total_valid, "invalid": total_invalid})

        all_results[bench_name] = results

    # 결과 출력
    print("\n" + "=" * 60)
    print("검증 결과")
    print("=" * 60)
    print(f"정상: {total_valid}개")
    print(f"손상: {total_invalid}개")

    if total_invalid > 0:
        print("\n손상된 데이터셋:")
        for bench_name, results in all_results.items():
            for r in results:
                if not r["valid"]:
                    print(f"  [{bench_name}] {Path(r['path']).name}: {r['error']}")
        sys.exit(1)
    else:
        print("\n모든 데이터셋이 정상입니다!")

    # 요약 저장
    summary_path = data_dir / "verify_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_valid": total_valid,
            "total_invalid": total_invalid,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"검증 결과 저장: {summary_path}")


if __name__ == "__main__":
    main()
