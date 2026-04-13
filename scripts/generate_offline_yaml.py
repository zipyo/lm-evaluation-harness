#!/usr/bin/env python3
"""
오프라인 환경용 태스크 YAML 생성 스크립트

Usage:
    python generate_offline_yaml.py --data-dir ./offline_datasets --output-dir ./offline_tasks
"""

import argparse
import os
from pathlib import Path


# MMLU 서브셋 목록
MMLU_SUBSETS = [
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
]

# KMMLU 서브셋 목록
KMMLU_SUBSETS = [
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
]


MMLU_TEMPLATE = '''# Auto-generated offline task for MMLU {subset}
task: {task_name}
dataset_path: {dataset_path}
dataset_kwargs:
  trust_remote_code: true
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: !function utils.mmlu_doc_to_text
doc_to_choice:
  - A
  - B
  - C
  - D
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
'''

KMMLU_TEMPLATE = '''# Auto-generated offline task for KMMLU {subset}
task: {task_name}
dataset_path: {dataset_path}
dataset_kwargs:
  trust_remote_code: true
test_split: test
fewshot_split: dev
output_type: multiple_choice
doc_to_text: !function utils.kmmlu_doc_to_text
doc_to_choice:
  - A
  - B
  - C
  - D
doc_to_target: !function utils.kmmlu_doc_to_target
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 2.0
'''

IFEVAL_TEMPLATE = '''# Auto-generated offline task for {name}
task: {task_name}
dataset_path: {dataset_path}
dataset_name: null
output_type: generate_until
test_split: train
num_fewshot: 0
doc_to_text: prompt
doc_to_target: 0
generation_kwargs:
  until: []
  do_sample: false
  temperature: 0.0
  max_gen_toks: 1280
process_results: !function utils.process_results
metric_list:
  - metric: prompt_level_strict_acc
    aggregation: mean
    higher_is_better: true
  - metric: inst_level_strict_acc
    aggregation: !function utils.agg_inst_level_acc
    higher_is_better: true
  - metric: prompt_level_loose_acc
    aggregation: mean
    higher_is_better: true
  - metric: inst_level_loose_acc
    aggregation: !function utils.agg_inst_level_acc
    higher_is_better: true
metadata:
  version: 1.0
'''

IFEVAL_CONFIGS = {
    "ifeval": {
        "name": "IFEval",
        "dir_name": "IFEval",
    },
}

UTILS_PY = '''# Auto-generated utils for offline tasks

def mmlu_doc_to_text(doc):
    """MMLU 문서를 텍스트로 변환"""
    question = doc["question"].strip()
    choices = doc["choices"]
    return f"{question}\\nA. {choices[0]}\\nB. {choices[1]}\\nC. {choices[2]}\\nD. {choices[3]}\\nAnswer:"


def kmmlu_doc_to_text(doc):
    """KMMLU 문서를 텍스트로 변환"""
    question = doc["question"].strip()
    return f"{question}\\nA. {doc['A']}\\nB. {doc['B']}\\nC. {doc['C']}\\nD. {doc['D']}\\n정답："


def kmmlu_doc_to_target(doc):
    """KMMLU 정답 추출 (1-indexed -> 0-indexed)"""
    return doc["answer"] - 1
'''


def generate_mmlu_yaml(subset: str, data_dir: Path, output_dir: Path):
    """MMLU 오프라인 YAML 생성"""
    task_name = f"mmlu_{subset}_offline"
    dataset_path = str(data_dir / "mmlu" / subset)

    yaml_content = MMLU_TEMPLATE.format(
        subset=subset,
        task_name=task_name,
        dataset_path=dataset_path
    )

    output_file = output_dir / "mmlu" / f"{task_name}.yaml"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml_content, encoding="utf-8")
    return task_name


def generate_kmmlu_yaml(subset: str, data_dir: Path, output_dir: Path):
    """KMMLU 오프라인 YAML 생성"""
    task_name = f"kmmlu_{subset.lower().replace('-', '_')}_offline"
    dataset_path = str(data_dir / "kmmlu" / subset)

    yaml_content = KMMLU_TEMPLATE.format(
        subset=subset,
        task_name=task_name,
        dataset_path=dataset_path
    )

    output_file = output_dir / "kmmlu" / f"{task_name}.yaml"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml_content, encoding="utf-8")
    return task_name


def generate_ifeval_yaml(name: str, data_dir: Path, output_dir: Path):
    """IFEval 오프라인 YAML 생성"""
    config = IFEVAL_CONFIGS[name]
    task_name = f"{name}_offline"
    dataset_path = str(data_dir / config["dir_name"])

    yaml_content = IFEVAL_TEMPLATE.format(
        name=config["name"],
        task_name=task_name,
        dataset_path=dataset_path
    )

    ifeval_dir = output_dir / "ifeval"
    ifeval_dir.mkdir(parents=True, exist_ok=True)

    output_file = ifeval_dir / f"{task_name}.yaml"
    output_file.write_text(yaml_content, encoding="utf-8")

    # utils.py 생성 (기존 ifeval utils를 재사용)
    utils_file = ifeval_dir / "utils.py"
    if not utils_file.exists():
        utils_file.write_text(
            "# IFEval offline utils — 기존 ifeval utils를 재사용\n"
            "from lm_eval.tasks.ifeval.utils import process_results, agg_inst_level_acc\n",
            encoding="utf-8",
        )

    return task_name


def generate_group_yaml(group_name: str, tasks: list, output_dir: Path):
    """그룹 YAML 생성"""
    tasks_str = "\n".join(f"  - {t}" for t in tasks)

    yaml_content = f'''# Auto-generated offline group
group: {group_name}
task:
{tasks_str}
aggregate_metric_list:
  - metric: acc
    weight_by_size: true
'''

    output_file = output_dir / f"_{group_name}.yaml"
    output_file.write_text(yaml_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="오프라인 태스크 YAML 생성")
    parser.add_argument("--data-dir", type=str, required=True, help="다운로드된 데이터셋 경로")
    parser.add_argument("--output-dir", type=str, required=True, help="YAML 출력 경로")
    parser.add_argument("--benchmark", type=str, choices=["all", "mmlu", "kmmlu", "ifeval"], default="all")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"데이터 경로: {data_dir}")
    print(f"출력 경로: {output_dir}")

    # utils.py 생성
    utils_file = output_dir / "utils.py"
    utils_file.write_text(UTILS_PY, encoding="utf-8")
    print(f"생성: {utils_file}")

    mmlu_tasks = []
    kmmlu_tasks = []

    # MMLU 생성
    if args.benchmark in ["all", "mmlu"]:
        print("\n=== MMLU 오프라인 YAML 생성 ===")
        for subset in MMLU_SUBSETS:
            subset_path = data_dir / "mmlu" / subset
            if subset_path.exists():
                task_name = generate_mmlu_yaml(subset, data_dir, output_dir)
                mmlu_tasks.append(task_name)
                print(f"  생성: {task_name}")
            else:
                print(f"  건너뜀 (데이터 없음): {subset}")

        if mmlu_tasks:
            generate_group_yaml("mmlu_offline", mmlu_tasks, output_dir)
            print(f"\n그룹 생성: mmlu_offline ({len(mmlu_tasks)}개 태스크)")

    # KMMLU 생성
    if args.benchmark in ["all", "kmmlu"]:
        print("\n=== KMMLU 오프라인 YAML 생성 ===")
        for subset in KMMLU_SUBSETS:
            subset_path = data_dir / "kmmlu" / subset
            if subset_path.exists():
                task_name = generate_kmmlu_yaml(subset, data_dir, output_dir)
                kmmlu_tasks.append(task_name)
                print(f"  생성: {task_name}")
            else:
                print(f"  건너뜀 (데이터 없음): {subset}")

        if kmmlu_tasks:
            generate_group_yaml("kmmlu_offline", kmmlu_tasks, output_dir)
            print(f"\n그룹 생성: kmmlu_offline ({len(kmmlu_tasks)}개 태스크)")

    # IFEval 생성
    ifeval_tasks = []
    if args.benchmark in ["all", "ifeval"]:
        print("\n=== IFEval 오프라인 YAML 생성 ===")
        for name, config in IFEVAL_CONFIGS.items():
            dataset_path = data_dir / config["dir_name"]
            if dataset_path.exists():
                task_name = generate_ifeval_yaml(name, data_dir, output_dir)
                ifeval_tasks.append(task_name)
                print(f"  생성: {task_name}")
            else:
                print(f"  건너뜀 (데이터 없음): {name} ({dataset_path})")

    print("\n" + "=" * 50)
    print("완료!")
    print(f"MMLU: {len(mmlu_tasks)}개")
    print(f"KMMLU: {len(kmmlu_tasks)}개")
    print(f"IFEval: {len(ifeval_tasks)}개")
    print("\n사용법:")
    print(f"  lm-eval run --tasks mmlu_offline,kmmlu_offline --include_path {output_dir} ...")


if __name__ == "__main__":
    main()
