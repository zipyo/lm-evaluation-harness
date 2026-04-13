# 오프라인 환경 평가 가이드

인터넷이 차단된 GPU 서버에서 lm-evaluation-harness로 LLM 평가를 수행하는 방법.

## 지원 벤치마크

| 벤치마크 | 다운로드 스크립트 | 오프라인 YAML |
|----------|------------------|--------------|
| MMLU (57개 과목) | `download_benchmarks.py --benchmark mmlu` | `generate_offline_yaml.py` |
| KMMLU (45개 과목) | `download_benchmarks.py --benchmark kmmlu` | `generate_offline_yaml.py` |
| KMMLU-HARD (45개 과목) | `download_benchmarks.py --benchmark kmmlu_hard` | `generate_offline_yaml.py` |
| KBL (68개 서브셋) | `download_benchmarks.py --benchmark kbl` | `generate_kbl_offline_yaml.py` |
| IFEval | `download_benchmarks.py --benchmark ifeval` | `lm_eval/tasks/ifeval/ifeval_offline.yaml` |
| IFEval-KO | 수동 복사 (HF 미공개) | `lm_eval/tasks/ifeval/ifeval_ko_offline.yaml` |

## 1단계: 데이터셋 다운로드 (온라인 환경)

```bash
# 전체 다운로드
python scripts/download_benchmarks.py --output-dir ./offline_datasets --benchmark all

# 개별 벤치마크
python scripts/download_benchmarks.py --output-dir ./offline_datasets --benchmark ifeval
python scripts/download_benchmarks.py --output-dir ./offline_datasets --benchmark mmlu
python scripts/download_benchmarks.py --output-dir ./offline_datasets --benchmark kbl

# 기업 프록시 환경
python scripts/download_benchmarks.py --output-dir ./offline_datasets --benchmark all \
    --proxy http://proxy:8080 --no-ssl
```

다운로드된 데이터셋은 Arrow 형식으로 `offline_datasets/`에 저장된다.

### 디렉토리 구조

```
offline_datasets/
├── mmlu/
│   ├── abstract_algebra/
│   ├── anatomy/
│   └── ...
├── kmmlu/
│   ├── Accounting/
│   └── ...
├── kbl/
│   ├── bar_exam_civil_2012/
│   └── ...
├── IFEval/
├── instruction-following-eval-ko/
└── download_summary.json
```

## 2단계: 오프라인 YAML 생성

MMLU/KMMLU는 서브셋이 많아 YAML을 동적으로 생성한다. IFEval은 이미 YAML이 포함되어 있어 이 단계가 불필요하다.

```bash
# MMLU/KMMLU 오프라인 YAML 생성
python scripts/generate_offline_yaml.py \
    --data-dir ./offline_datasets \
    --output-dir ./offline_tasks

# KBL 오프라인 YAML 생성
python scripts/generate_kbl_offline_yaml.py \
    --data-dir ./offline_datasets/kbl \
    --output-dir ./lm_eval/tasks/kbl_offline
```

## 3단계: 오프라인 환경으로 전송

```bash
# 압축
tar -czf lm_eval_offline.tar.gz offline_datasets/ offline_tasks/

# GPU 서버로 전송
scp lm_eval_offline.tar.gz user@gpu-server:/workspace/

# GPU 서버에서 압축 해제
ssh user@gpu-server
cd /workspace/lm-evaluation-harness
tar -xzf /workspace/lm_eval_offline.tar.gz
```

## 4단계: 오프라인 평가 실행

### 환경 변수 설정

```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### vLLM으로 실행 (GPU, 권장)

```bash
# IFEval-KO
lm-eval run --model vllm \
    --model_args pretrained=/path/to/model,tensor_parallel_size=2,dtype=bfloat16,max_model_len=8192 \
    --tasks ifeval_ko_offline \
    --batch_size auto \
    --output_path ./results

# IFEval (영어)
lm-eval run --model vllm \
    --model_args pretrained=/path/to/model,tensor_parallel_size=2,dtype=bfloat16 \
    --tasks ifeval_offline \
    --batch_size auto \
    --output_path ./results

# 생성된 오프라인 YAML 사용 (MMLU/KMMLU)
lm-eval run --model vllm \
    --model_args pretrained=/path/to/model,tensor_parallel_size=4,dtype=bfloat16 \
    --tasks mmlu_offline,kmmlu_offline \
    --include_path ./offline_tasks \
    --batch_size auto \
    --output_path ./results
```

### OpenAI API 호환 서버 사용

vLLM을 API 서버로 띄운 후 연결할 수도 있다.

```bash
# 1. vLLM API 서버 시작
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --tensor-parallel-size 4

# 2. lm-eval에서 API로 연결
lm-eval run --model local-chat-completions \
    --model_args model=my-model,base_url=http://localhost:8000/v1 \
    --tasks ifeval_ko_offline \
    --batch_size auto \
    --output_path ./results
```

### Chat 모델 (Instruct) 사용 시

```bash
lm-eval run --model vllm \
    --model_args pretrained=/path/to/model,tensor_parallel_size=2,dtype=bfloat16 \
    --tasks ifeval_ko_offline \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size auto \
    --output_path ./results
```

## 문제 해결

### "Dataset not found" 오류

```bash
# 환경 변수 확인
env | grep HF_

# 데이터셋 경로 확인
ls offline_datasets/
python -c "from datasets import load_from_disk; print(load_from_disk('offline_datasets/IFEval'))"
```

### "Connection error" 오류

오프라인 환경 변수가 설정되지 않은 경우 발생한다.

```bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### vLLM 메모리 부족

```bash
# max_model_len을 줄이거나 GPU 수를 늘린다
--model_args pretrained=/path/to/model,tensor_parallel_size=4,max_model_len=4096,gpu_memory_utilization=0.9
```
