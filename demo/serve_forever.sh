#!/bin/bash
# Keep one inference server alive: restart it 10 s after any exit.
#   demo/serve_forever.sh 151 8765 2 [lora|zeroshot]     (labels, port, GPU, style)
cd "$(dirname "$0")/.."
export CLINC_API_KEY=$(cat demo/.api_key)
STYLE=${4:-lora}
LOG=logs/demo_server_${STYLE}_$1.log
while true; do
  echo "[$(date '+%F %T')] starting server ($STYLE, $1 labels, port $2, GPU $3)" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$3 /mnt/ssd2/henry/conda_envs/trl/bin/python demo/server.py --style $STYLE --labels $1 --port $2 >> $LOG 2>&1
  echo "[$(date '+%F %T')] server exited with code $?, restarting in 10 s" | tee -a $LOG
  sleep 10
done
