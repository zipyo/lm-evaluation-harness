#!/usr/bin/env python3
"""
벤치마크 데이터셋 다운로드 스크립트 (MMLU, KMMLU, KBL, IFEval)

save_to_disk() Arrow 형식으로 저장하여 오프라인 환경에서 사용 가능하도록 함.

Usage:
    # 전체 다운로드
    python download_benchmarks.py --output-dir ./offline_datasets --benchmark all

    # 개별 벤치마크 다운로드
    python download_benchmarks.py --output-dir ./offline_datasets --benchmark mmlu
    python download_benchmarks.py --output-dir ./offline_datasets --benchmark kmmlu
    python download_benchmarks.py --output-dir ./offline_datasets --benchmark kbl
    python download_benchmarks.py --output-dir ./offline_datasets --benchmark ifeval

    # 서브셋 목록 확인
    python download_benchmarks.py --list-subsets

    # SSL 인증서 검증 비활성화 (기업 프록시 환경)
    python download_benchmarks.py --output-dir ./offline_datasets --benchmark kbl --no-ssl

    # 프록시 + SSL 비활성화
    python download_benchmarks.py --output-dir ./offline_datasets --benchmark kbl --proxy http://proxy:8080 --no-ssl
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
    from tqdm import tqdm
    import urllib3
except ImportError:
    print("Required packages not found. Please install:")
    print("pip install datasets tqdm")
    sys.exit(1)

# 로거 초기화 (함수에서 사용하기 전에 정의)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_proxy(proxy_url: str):
    """프록시 설정 (기업 환경용)"""
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    logger.info(f"프록시 설정됨: {proxy_url}")


def disable_ssl_verification():
    """SSL 인증서 검증 비활성화 (기업 프록시 환경용)"""
    logger.warning("SSL 인증서 검증 비활성화됨 - 신뢰할 수 있는 네트워크에서만 사용하세요")

    # HuggingFace Hub SSL 검증 비활성화 환경변수
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

    # urllib3 경고 비활성화
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================
# 벤치마크 정의
# ============================================================

BENCHMARKS = {
    "mmlu": {
        "dataset_path": "cais/mmlu",
        "description": "Massive Multitask Language Understanding (57 subjects)",
        "subsets": [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history", "human_aging",
            "human_sexuality", "international_law", "jurisprudence",
            "logical_fallacies", "machine_learning", "management", "marketing",
            "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions",
        ],
    },
    "kmmlu": {
        "dataset_path": "HAERAE-HUB/KMMLU",
        "description": "Korean MMLU (45 subjects)",
        "subsets": [
            "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance",
            "Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering",
            "Computer-Science", "Construction", "Criminal-Law", "Ecology", "Economics",
            "Education", "Electrical-Engineering", "Electronics-Engineering",
            "Energy-Management", "Environmental-Science", "Fashion", "Food-Processing",
            "Gas-Technology-and-Engineering", "Geomatics", "Health", "Industrial-Engineer",
            "Information-Technology", "Interior-Architecture-and-Design", "Korean-History",
            "Law", "Machine-Design-and-Manufacturing", "Management", "Maritime-Engineering",
            "Marketing", "Materials-Engineering", "Math", "Mechanical-Engineering",
            "Nondestructive-Testing", "Patent", "Political-Science-and-Sociology",
            "Psychology", "Public-Safety", "Railway-and-Automotive-Engineering",
            "Real-Estate", "Refrigerating-Machinery", "Social-Welfare", "Taxation",
            "Telecommunications-and-Wireless-Technology",
        ],
    },
    "kmmlu_hard": {
        "dataset_path": "HAERAE-HUB/KMMLU-HARD",
        "description": "Korean MMLU Hard (45 subjects)",
        "subsets": [
            "accounting", "agricultural_sciences", "aviation_engineering_and_maintenance",
            "biology", "chemical_engineering", "chemistry", "civil_engineering",
            "computer_science", "construction", "criminal_law", "ecology", "economics",
            "education", "electrical_engineering", "electronics_engineering",
            "energy_management", "environmental_science", "fashion", "food_processing",
            "gas_technology_and_engineering", "geomatics", "health", "industrial_engineer",
            "information_technology", "interior_architecture_and_design", "korean_history",
            "law", "machine_design_and_manufacturing", "management", "maritime_engineering",
            "marketing", "materials_engineering", "math", "mechanical_engineering",
            "nondestructive_testing", "patent", "political_science_and_sociology",
            "psychology", "public_safety", "railway_and_automotive_engineering",
            "real_estate", "refrigerating_machinery", "social_welfare", "taxation",
            "telecommunications_and_wireless_technology",
        ],
    },
    "ifeval": {
        "dataset_path": "google/IFEval",
        "description": "Instruction Following Eval",
        "subsets": [],
    },
    "kbl": {
        "dataset_path": "lbox/kbl",
        "description": "Korean Benchmark for Legal Language Understanding (68 subsets)",
        "subsets": (
            # Knowledge (7)
            [
                "kbl_knowledge_common_legal_mistake_qa",
                "kbl_knowledge_common_legal_mistake_qa_reasoning",
                "kbl_knowledge_legal_concept_qa",
                "kbl_knowledge_offense_component_qa",
                "kbl_knowledge_query_and_statute_matching_qa",
                "kbl_knowledge_statute_hallucination_qa",
                "kbl_knowledge_statute_number_and_content_matching_qa",
            ] +
            # Reasoning (4)
            [
                "kbl_reasoning_case_relevance_qa_p",
                "kbl_reasoning_case_relevance_qa_q",
                "kbl_reasoning_causal_reasoning_qa",
                "kbl_reasoning_statement_consistency_qa",
            ] +
            # Bar exam
            [f"bar_exam_civil_{year}" for year in range(2012, 2026)] +
            [f"bar_exam_criminal_{year}" for year in range(2012, 2026)] +
            [f"bar_exam_public_{year}" for year in range(2012, 2026)] +
            [f"bar_exam_responsibility_{year}" for year in range(2010, 2024)]
        ),
    },
}


def download_and_save(
    benchmark: str,
    subset: str,
    output_dir: Path,
    dataset_path: str,
) -> dict:
    """단일 서브셋 다운로드 및 저장"""
    save_path = output_dir / benchmark / subset
    result = {
        "benchmark": benchmark,
        "subset": subset,
        "save_path": str(save_path),
        "success": False,
        "splits": [],
        "error": None,
    }

    try:
        logger.info(f"다운로드 중: {dataset_path} - {subset}")

        # HuggingFace에서 다운로드
        ds = load_dataset(dataset_path, subset, trust_remote_code=True)

        # Arrow 형식으로 저장
        ds.save_to_disk(str(save_path))

        result["success"] = True
        result["splits"] = list(ds.keys())
        result["num_examples"] = {split: len(ds[split]) for split in ds.keys()}

        logger.info(f"✓ 저장 완료: {save_path}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"✗ 실패: {subset} - {e}")

    return result


def download_single_dataset(
    benchmark: str,
    output_dir: Path,
    dataset_path: str,
) -> dict:
    """서브셋 없는 단일 데이터셋 다운로드 및 저장"""
    save_path = output_dir / benchmark
    result = {
        "benchmark": benchmark,
        "subset": None,
        "save_path": str(save_path),
        "success": False,
        "splits": [],
        "error": None,
    }

    try:
        logger.info(f"다운로드 중: {dataset_path}")
        ds = load_dataset(dataset_path)
        ds.save_to_disk(str(save_path))

        result["success"] = True
        result["splits"] = list(ds.keys())
        result["num_examples"] = {split: len(ds[split]) for split in ds.keys()}

        logger.info(f"✓ 저장 완료: {save_path}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"✗ 실패: {benchmark} - {e}")

    return result


def download_benchmark(benchmark: str, output_dir: Path) -> List[dict]:
    """벤치마크 전체 다운로드"""
    if benchmark not in BENCHMARKS:
        logger.error(f"알 수 없는 벤치마크: {benchmark}")
        return []

    config = BENCHMARKS[benchmark]
    dataset_path = config["dataset_path"]
    subsets = config["subsets"]

    logger.info(f"=== {benchmark.upper()} 다운로드 시작 ===")
    logger.info(f"데이터셋: {dataset_path}")

    # 서브셋 없는 단일 데이터셋
    if not subsets:
        result = download_single_dataset(benchmark, output_dir, dataset_path)
        return [result]

    logger.info(f"서브셋 수: {len(subsets)}")

    results = []
    success_count = 0
    fail_count = 0

    with tqdm(total=len(subsets), desc=f"{benchmark} 다운로드") as pbar:
        for subset in subsets:
            result = download_and_save(benchmark, subset, output_dir, dataset_path)
            results.append(result)

            if result["success"]:
                success_count += 1
            else:
                fail_count += 1

            pbar.update(1)
            pbar.set_postfix({"success": success_count, "fail": fail_count})

    return results


def list_subsets():
    """모든 서브셋 목록 출력"""
    print("\n" + "=" * 60)
    print("사용 가능한 벤치마크 및 서브셋")
    print("=" * 60)

    total = 0
    for name, config in BENCHMARKS.items():
        subsets = config["subsets"]
        total += len(subsets)
        print(f"\n[{name.upper()}] - {config['description']}")
        print(f"  데이터셋: {config['dataset_path']}")
        print(f"  서브셋 수: {len(subsets)}")

        # 처음 5개와 마지막 2개만 표시
        if len(subsets) > 10:
            for s in subsets[:5]:
                print(f"    - {s}")
            print(f"    ... ({len(subsets) - 7}개 생략)")
            for s in subsets[-2:]:
                print(f"    - {s}")
        else:
            for s in subsets:
                print(f"    - {s}")

    print(f"\n총 서브셋 수: {total}개")


def main():
    parser = argparse.ArgumentParser(
        description="벤치마크 데이터셋을 Arrow 형식으로 다운로드 (오프라인 환경용)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="저장 디렉토리 경로"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "mmlu", "kmmlu", "kmmlu_hard", "kbl", "ifeval"],
        default="all",
        help="다운로드할 벤치마크 (기본값: all)"
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
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="SSL 인증서 검증 비활성화 (기업 프록시 환경용)"
    )
    parser.add_argument(
        "--proxy",
        type=str,
        help="프록시 URL (예: http://proxy.company.com:8080)"
    )

    args = parser.parse_args()

    # 프록시 설정
    if args.proxy:
        setup_proxy(args.proxy)

    # SSL 비활성화 (프록시 환경)
    if args.no_ssl:
        disable_ssl_verification()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 서브셋 목록 출력
    if args.list_subsets:
        list_subsets()
        return

    # --output-dir 필수 체크
    if not args.output_dir:
        parser.error("--output-dir is required when downloading datasets")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"출력 디렉토리: {output_dir}")

    # 다운로드할 벤치마크 결정
    if args.benchmark == "all":
        benchmarks = list(BENCHMARKS.keys())
    else:
        benchmarks = [args.benchmark]

    # 다운로드 실행
    all_results = {}
    total_success = 0
    total_fail = 0

    for benchmark in benchmarks:
        results = download_benchmark(benchmark, output_dir)
        all_results[benchmark] = results

        success = sum(1 for r in results if r["success"])
        fail = sum(1 for r in results if not r["success"])
        total_success += success
        total_fail += fail

    # 결과 요약 저장
    summary = {
        "output_dir": str(output_dir),
        "benchmarks": benchmarks,
        "total_success": total_success,
        "total_failed": total_fail,
        "results": all_results,
    }

    summary_path = output_dir / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 콘솔 출력
    print("\n" + "=" * 60)
    print("다운로드 완료")
    print("=" * 60)
    print(f"성공: {total_success}개")
    print(f"실패: {total_fail}개")
    print(f"저장 경로: {output_dir}")
    print(f"요약 파일: {summary_path}")

    if total_fail > 0:
        print("\n실패한 서브셋:")
        for benchmark, results in all_results.items():
            for r in results:
                if not r["success"]:
                    print(f"  [{benchmark}] {r['subset']}: {r['error']}")

    print("\n오프라인 환경에서 사용하려면:")
    print("  export HF_DATASETS_OFFLINE=1")
    print("  export HF_HUB_OFFLINE=1")
    print("  export TRANSFORMERS_OFFLINE=1")


if __name__ == "__main__":
    main()
