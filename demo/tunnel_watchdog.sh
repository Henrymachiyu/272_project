#!/bin/bash
# Every 60 s: if a tunnel process is gone, start a new one (new URL) and record it in demo/tunnel_urls.txt.
CF=~/tools/cloudflared/cloudflared
cd "$(dirname "$0")/.."
while true; do
  for spec in "151 8765" "150 8766" "zs151 8767" "zs150 8768"; do
    set -- $spec
    if ! pgrep -f "cloudflared tunnel --no-autoupdate --url http://localhost:$2" > /dev/null; then
      tmux kill-session -t tunnel$1 2>/dev/null
      : > ~/tools/cloudflared/tunnel$1.log
      tmux new-session -d -s tunnel$1 "$CF tunnel --no-autoupdate --url http://localhost:$2 2>&1 | tee ~/tools/cloudflared/tunnel$1.log"
      sleep 15
      echo "[$(date '+%F %T')] $1 labels: $(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/tools/cloudflared/tunnel$1.log | head -1)  (restarted)" >> demo/tunnel_urls.txt
    fi
  done
  sleep 60
done
