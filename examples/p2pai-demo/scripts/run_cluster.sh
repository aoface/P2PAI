#!/usr/bin/env bash
# 在本机以三节点拓扑跑一次完整的 P2P 训推一体演示.
# 用法:
#   bash scripts/run_cluster.sh                  # 默认 80s 总时长, 8 个训练步
#   RUN_TIME=120 MAX_STEPS=16 bash scripts/run_cluster.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_TIME=${RUN_TIME:-80}
MAX_STEPS=${MAX_STEPS:-8}
LOG_LEVEL=${LOG_LEVEL:-INFO}

mkdir -p logs
rm -rf .p2pai-demo

echo "[run_cluster] starting node2 (worker)..."
python3 -m p2pai_demo.cli -c configs/node2.toml --run-time "$RUN_TIME" \
  --log-level "$LOG_LEVEL" > logs/node2.log 2>&1 &
P2=$!

echo "[run_cluster] starting node3 (worker)..."
python3 -m p2pai_demo.cli -c configs/node3.toml --run-time "$RUN_TIME" \
  --log-level "$LOG_LEVEL" > logs/node3.log 2>&1 &
P3=$!

sleep 8  # 给 worker 节点时间加载模型并连接

echo "[run_cluster] starting node1 (coordinator + task submitter)..."
python3 -m p2pai_demo.cli -c configs/node1.toml --submit \
  --prompts data/sample_prompts.txt \
  --max-steps "$MAX_STEPS" --min-group 1 \
  --run-time "$RUN_TIME" --log-level "$LOG_LEVEL" 2>&1 | tee logs/node1.log
P1=$!

wait $P2 $P3 || true
echo "[run_cluster] done; see logs/node{1,2,3}.log"
