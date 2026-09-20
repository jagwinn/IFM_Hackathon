#!/bin/sh
# Start llama-server and publish the hardware it runs on at GET /device.json, which the relay reads to
# label each model ("Local · RTX 3070"). All arguments are passed to llama-server.
set -e
info=/srv/relay-info
mkdir -p "$info"

gpus=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
fi

if [ -n "$gpus" ]; then
  # nvidia-smi numbers only the GPUs this container can see, from 0. RELAY_GPU_INDEXES (set by compose.gpu.yaml
  # next to device_ids) gives their host indexes in the same order; images often leave NVIDIA_VISIBLE_DEVICES=all.
  echo "$gpus" | awk -v visible="${RELAY_GPU_INDEXES:-${NVIDIA_VISIBLE_DEVICES:-all}}" '
    BEGIN { n = split(visible, host, ","); numeric = n > 0; for (i = 1; i <= n; i++) if (host[i] !~ /^[0-9]+$/) numeric = 0 }
    { gsub(/"/, "", $0); idx = (numeric && NR <= n) ? host[NR] : NR - 1; items = items (NR > 1 ? "," : "") "{\"index\":" idx ",\"name\":\"" $0 "\"}" }
    END { printf "{\"kind\":\"gpu\",\"gpus\":[%s]}\n", items }' > "$info/device.json"
else
  cpu=$(grep -m1 -E '^(model name|Hardware)' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//; s/"//g')
  printf '{"kind":"cpu","name":"%s"}\n' "${cpu:-CPU}" > "$info/device.json"
fi

exec llama-server --path "$info" "$@"
