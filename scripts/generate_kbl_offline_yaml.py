#!/usr/bin/env python3
"""
KBL 오프라인 태스크 YAML 생성 스크립트

다운로드된 Arrow 데이터셋을 기반으로 lm-eval 오프라인 태스크 YAML을 생성합니다.

Usage:
    python generate_kbl_offline_yaml.py --data-dir /path/to/offline_datasets/kbl --output-dir ./lm_eval/tasks/kbl_offline
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

# KBL 서브셋 구조
KBL_STRUCTURE = {
    "bar_exam": {
        "civil": [f"bar_exam_civil_{year}" for year in range(2012, 2026)],
        "criminal": [f"bar_exam_criminal_{year}" for year in range(2012, 2025)],
        "public": [f"bar_exam_public_{year}" for year in range(2012, 2025)],
        "responsibility": [f"bar_exam_responsibility_{year}" for year in range(2010, 2024)],
    },
    "knowledge": {
        "_": [
            "kbl_knowledge_common_legal_mistake_qa",
            "kbl_knowledge_common_legal_mistake_qa_reasoning",
            "kbl_knowledge_legal_concept_qa",
            "kbl_knowledge_offense_component_qa",
            "kbl_knowledge_query_statute_matching_qa",
            "kbl_knowledge_statute_hallucination_qa",
            "kbl_knowledge_statute_number_and_content_matching_qa",
        ]
    },
    "reasoning": {
        "_": [
            "kbl_reasoning_case_relevance_qa_p",
            "kbl_reasoning_case_relevance_qa_q",
            "kbl_reasoning_causal_reasoning",
            "kbl_reasoning_statement_consistency_qa",
        ]
    },
}

# 태스크 타입별 doc_to_text 템플릿
DOC_TO_TEXT_TEMPLATES = {
    "bar_exam": '''### 질문: {{question}}

다음 각 선택지를 읽고 A, B, C, D, E 중 하나를 선택하여 '답변: A' 와 같이 단답식으로 답해 주세요.

A. {{A}}

B. {{B}}

C. {{C}}

D. {{D}}

E. {{E}}

### 답변:''',

    "knowledge_default": '''### 질문: {{question}}
A. {{A}}
B. {{B}}
C. {{C}}
D. {{D}}
E. {{E}}
'A', 'B', 'C', 'D', 'E' 중 하나를 선택하여 '답변: A' 와 같이 단답식으로 답해 주세요.''',

    "reasoning_default": '''### 질문: {{question}}
A. {{A}}
B. {{B}}
'A', 'B' 중 하나를 선택하여 '답변: A'과 같이 단답식으로 답해주세요.''',
}


def generate_base_yaml(category: str, output_dir: Path, doc_to_text: str, doc_to_target: str = "{{label}}"):
    """Base YAML 생성"""
    base_content = f'''tag:
  - kbl_offline
  - kbl_offline_{category}
description: 'KBL {category} 평가 (오프라인 버전)'
dataset_name: null
test_split: test
output_type: generate_until
doc_to_text: |
  {doc_to_text}
doc_to_target: "{doc_to_target}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
filter_list:
  - name: get-answer
    filter:
    - function: regex
      regex_pattern: "([A-E]).*"
    - function: take_first
'''
    base_path = output_dir / f"_base_{category}_yaml"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    with open(base_path, "w", encoding="utf-8") as f:
        f.write(base_content)
    return base_path


def generate_task_yaml(
    subset: str,
    data_dir: str,
    output_path: Path,
    base_yaml: str,
    category: str,
):
    """개별 태스크 YAML 생성"""
    task_name = f"kbl_offline_{subset.replace('-', '_')}"
    dataset_path = f"{data_dir}/{subset}"

    content = f'''task: {task_name}
dataset_path: {dataset_path}
include: {base_yaml}
'''

    # 태스크별 태그 추가
    if "civil" in subset:
        content += '''tag:
  - kbl_offline
  - kbl_offline_bar_exam
  - kbl_offline_bar_exam_civil
'''
    elif "criminal" in subset:
        content += '''tag:
  - kbl_offline
  - kbl_offline_bar_exam
  - kbl_offline_bar_exam_criminal
'''
    elif "public" in subset:
        content += '''tag:
  - kbl_offline
  - kbl_offline_bar_exam
  - kbl_offline_bar_exam_public
'''
    elif "responsibility" in subset:
        content += '''tag:
  - kbl_offline
  - kbl_offline_bar_exam
  - kbl_offline_bar_exam_responsibility
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return task_name


def main():
    parser = argparse.ArgumentParser(
        description="KBL 오프라인 태스크 YAML 생성"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="다운로드된 KBL Arrow 데이터셋 경로 (예: /path/to/offline_datasets/kbl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lm_eval/tasks/kbl_offline",
        help="생성된 YAML 저장 경로"
    )

    args = parser.parse_args()

    data_dir = args.data_dir.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"데이터 경로: {data_dir}")
    print(f"출력 경로: {output_dir}")

    generated_tasks = []

    # Bar exam 태스크 생성
    bar_exam_dir = output_dir / "bar_exam"
    bar_exam_base = generate_base_yaml(
        "bar_exam",
        bar_exam_dir,
        DOC_TO_TEXT_TEMPLATES["bar_exam"],
        "gt"
    )
    print(f"생성됨: {bar_exam_base}")

    for subcategory, subsets in KBL_STRUCTURE["bar_exam"].items():
        subcat_dir = bar_exam_dir / subcategory
        for subset in subsets:
            yaml_path = subcat_dir / f"kbl_offline_{subset}.yaml"
            task_name = generate_task_yaml(
                subset,
                data_dir,
                yaml_path,
                "_base_bar_exam_yaml",
                "bar_exam"
            )
            generated_tasks.append(task_name)
            print(f"생성됨: {yaml_path}")

    # Knowledge 태스크 생성
    knowledge_dir = output_dir / "knowledge"
    knowledge_base = generate_base_yaml(
        "knowledge",
        knowledge_dir,
        DOC_TO_TEXT_TEMPLATES["knowledge_default"],
        "{{label}}"
    )
    print(f"생성됨: {knowledge_base}")

    for subset in KBL_STRUCTURE["knowledge"]["_"]:
        yaml_path = knowledge_dir / f"kbl_offline_{subset}.yaml"
        task_name = generate_task_yaml(
            subset,
            data_dir,
            yaml_path,
            "_base_knowledge_yaml",
            "knowledge"
        )
        generated_tasks.append(task_name)
        print(f"생성됨: {yaml_path}")

    # Reasoning 태스크 생성
    reasoning_dir = output_dir / "reasoning"
    reasoning_base = generate_base_yaml(
        "reasoning",
        reasoning_dir,
        DOC_TO_TEXT_TEMPLATES["reasoning_default"],
        "{{label}}"
    )
    print(f"생성됨: {reasoning_base}")

    for subset in KBL_STRUCTURE["reasoning"]["_"]:
        yaml_path = reasoning_dir / f"kbl_offline_{subset}.yaml"
        task_name = generate_task_yaml(
            subset,
            data_dir,
            yaml_path,
            "_base_reasoning_yaml",
            "reasoning"
        )
        generated_tasks.append(task_name)
        print(f"생성됨: {yaml_path}")

    # README 생성
    readme_content = f'''# KBL Offline Tasks

오프라인 환경에서 KBL (Korean Benchmark for Legal Language Understanding) 평가를 위한 태스크입니다.

## 데이터 경로 설정

현재 설정된 데이터 경로: `{data_dir}`

이 경로가 올바르지 않다면 `generate_kbl_offline_yaml.py` 스크립트를 다시 실행하세요:

```bash
python scripts/generate_kbl_offline_yaml.py --data-dir /your/actual/path --output-dir ./lm_eval/tasks/kbl_offline
```

## 태그

- `kbl_offline` - 모든 오프라인 KBL 태스크
- `kbl_offline_bar_exam` - 변호사 시험 전체
- `kbl_offline_bar_exam_civil` - 민법
- `kbl_offline_bar_exam_criminal` - 형법
- `kbl_offline_bar_exam_public` - 공법
- `kbl_offline_bar_exam_responsibility` - 직무윤리
- `kbl_offline_knowledge` - 법률 지식
- `kbl_offline_reasoning` - 법률 추론

## 실행 예시

```bash
# 환경변수 설정
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 전체 실행
lm-eval run \\
  --model vllm \\
  --model_args pretrained=/path/to/model,tensor_parallel_size=4 \\
  --tasks kbl_offline \\
  --batch_size auto \\
  --output_path ./results

# 특정 카테고리만 실행
lm-eval run \\
  --model vllm \\
  --model_args pretrained=/path/to/model \\
  --tasks kbl_offline_bar_exam_civil \\
  --batch_size auto
```

## 생성된 태스크 목록

{chr(10).join(f"- {task}" for task in sorted(generated_tasks))}
'''

    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"생성됨: {readme_path}")

    print("\n" + "=" * 60)
    print(f"총 {len(generated_tasks)}개 태스크 YAML 생성 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
