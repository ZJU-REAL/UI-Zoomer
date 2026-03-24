#!/bin/bash

echo "🧹 [1/4] Cleaning up GPU memory..."
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep vllm | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 3
echo "✅ Cleanup complete"

# Path Configuration

MODEL_PATH="/root/autodl-tmp/models/UI-Venus-7B"
IMG_PATH="/root/autodl-tmp/benchmark/ScreenSpot-Pro/images"
TEST_PATH="/root/autodl-tmp/benchmark/ScreenSpot-Pro/annotations"
LOG_DIR="/root/autodl-tmp/logs"
mkdir -p $LOG_DIR

# Core Configuration
SCRIPT_NAME="UI-Zoomer/uizoomer.py"
SAMPLES=8
THRESHOLD=1.5

# 🎯 Set Sigma Value
SIGMA=2.5
EXP_TAG="gaussian_sigma_${SIGMA}"

echo "🚀 [2/4] Starting Experiment: Sigma = ${SIGMA}"
echo "   - Config: Density Filter + Gaussian Variance * ${SIGMA}"

# Data Parallelism(4 GPUs)
for i in {0..3}; do
    JSON_PATH="${LOG_DIR}/${EXP_TAG}_s${SAMPLES}_part${i}.json"
    TEXT_LOG="${LOG_DIR}/run_log_${EXP_TAG}_s${SAMPLES}_part${i}.txt"
    
    echo "▶️  GPU $i start -> Chunk $i -> Log: $TEXT_LOG"
    
    CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME \
       --model_name_or_path "$MODEL_PATH" \
       --screenspot_imgs "$IMG_PATH" \
       --screenspot_test "$TEST_PATH" \
       --task "all" \
       --log_path "$JSON_PATH" \
       --chunk_id $i \
       --num_chunks 4 \
       --num_samples $SAMPLES \
       --gating_threshold $THRESHOLD \
       --sigma_scale $SIGMA \
       > "$TEXT_LOG" 2>&1 &
done

echo ""
echo "⏳ [3/4] Tasks started, currently computing..."
wait
echo "✅ Tasks completed
