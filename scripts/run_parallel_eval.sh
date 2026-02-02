#!/bin/bash
# 37B 모델 4개를 H100 8장에서 병렬 평가하는 스크립트
#
# Usage:
#   ./scripts/run_parallel_eval.sh
#
# 설정:
#   - 각 모델당 2 GPU (tensor_parallel_size=2)
#   - 4개 모델 동시 실행

set -e

# ============================================================
# 설정 (수정 필요)
# ============================================================

# 모델 경로
MODEL1="/path/to/model1"
MODEL2="/path/to/model2"
MODEL3="/path/to/model3"
MODEL4="/path/to/model4"

# 평가 태스크 (쉼표로 구분)
TASKS="kbl"  # 또는 "mmlu,kmmlu,kbl"

# 출력 디렉토리
OUTPUT_DIR="./results"

# Chat 모델 옵션 (Instruct/Chat 모델이면 true)
USE_CHAT_TEMPLATE=true

# Fewshot 설정 (MMLU/KMMLU: 5, KBL: 0)
NUM_FEWSHOT=0  # KBL은 0, MMLU/KMMLU는 5 권장

# 최대 시퀀스 길이 (메모리 부족 시 줄이기)
MAX_MODEL_LEN=8192  # 평가용으로 충분한 길이

# 데이터셋 경로 (오프라인 모드 시)
# DATASET_PATH="./offline_datasets"

# ============================================================
# 환경 설정
# ============================================================

mkdir -p "$OUTPUT_DIR" logs

# 오프라인 모드 설정 (필요시 주석 해제)
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Chat template 옵션 설정
CHAT_OPTS=""
if [ "$USE_CHAT_TEMPLATE" = true ]; then
    CHAT_OPTS="--apply_chat_template --fewshot_as_multiturn"
fi

# Fewshot 옵션 설정
FEWSHOT_OPTS=""
if [ "$NUM_FEWSHOT" -gt 0 ]; then
    FEWSHOT_OPTS="--num_fewshot $NUM_FEWSHOT"
fi

# ============================================================
# 모델 평가 실행
# ============================================================

echo "========================================"
echo "4개 모델 병렬 평가 시작"
echo "========================================"
echo "태스크: $TASKS"
echo "출력: $OUTPUT_DIR"
echo ""

# 모델 1: GPU 0,1
echo "[1/4] 모델 1 시작 (GPU 0,1)"
CUDA_VISIBLE_DEVICES=0,1 lm-eval run \
    --model vllm \
    --model_args pretrained=$MODEL1,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=$MAX_MODEL_LEN \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/model1" \
    $CHAT_OPTS $FEWSHOT_OPTS \
    &> logs/model1.log &
PID1=$!

# 모델 2: GPU 2,3
echo "[2/4] 모델 2 시작 (GPU 2,3)"
CUDA_VISIBLE_DEVICES=2,3 lm-eval run \
    --model vllm \
    --model_args pretrained=$MODEL2,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=$MAX_MODEL_LEN \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/model2" \
    $CHAT_OPTS $FEWSHOT_OPTS \
    &> logs/model2.log &
PID2=$!

# 모델 3: GPU 4,5
echo "[3/4] 모델 3 시작 (GPU 4,5)"
CUDA_VISIBLE_DEVICES=4,5 lm-eval run \
    --model vllm \
    --model_args pretrained=$MODEL3,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=$MAX_MODEL_LEN \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/model3" \
    $CHAT_OPTS $FEWSHOT_OPTS \
    &> logs/model3.log &
PID3=$!

# 모델 4: GPU 6,7
echo "[4/4] 모델 4 시작 (GPU 6,7)"
CUDA_VISIBLE_DEVICES=6,7 lm-eval run \
    --model vllm \
    --model_args pretrained=$MODEL4,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=$MAX_MODEL_LEN \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUTPUT_DIR/model4" \
    $CHAT_OPTS $FEWSHOT_OPTS \
    &> logs/model4.log &
PID4=$!

echo ""
echo "========================================"
echo "모든 모델 실행됨"
echo "========================================"
echo "로그 확인: tail -f logs/model*.log"
echo "GPU 확인: watch -n 1 nvidia-smi"
echo ""

# 모든 프로세스 대기
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "========================================"
echo "모든 평가 완료"
echo "========================================"
echo "결과: $OUTPUT_DIR"
