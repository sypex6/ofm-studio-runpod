#!/bin/bash
set -euo pipefail
cd /opt/studio-worker/serverless
test -d /runpod-volume || { echo 'Attach a Runpod Network Volume at /runpod-volume'; exit 1; }
test -n "${STUDIO_HOST:-}" || { echo 'Set STUDIO_HOST to the Studio hostname'; exit 1; }
df -h /runpod-volume /tmp
profiles="${WORKFLOW_PROFILES:-animate-ki,animate-wrapper}"
if [ "${DOWNLOAD_MODELS:-0}" = '1' ]; then
    python prepare_models.py --profiles "$profiles"
else
    python prepare_models.py --check --profiles "$profiles"
fi
cat > /opt/ComfyUI/extra_model_paths.yaml <<'YAML'
network_volume:
  base_path: /runpod-volume/models
  diffusion_models: diffusion_models
  unet: diffusion_models
  vae: vae
  text_encoders: text_encoders
  clip: text_encoders
  clip_vision: clip_vision
  loras: loras
  controlnet: controlnet
  detection: detection
YAML
python /opt/ComfyUI/main.py --listen 127.0.0.1 --port 8188 \
    --disable-auto-launch --preview-method none --use-pytorch-cross-attention > /tmp/comfyui.log 2>&1 &
comfy_pid=$!
worker_pid=''
cleanup() {
    if [ -n "$worker_pid" ]; then kill "$worker_pid" 2>/dev/null || true; fi
    kill "$comfy_pid" 2>/dev/null || true
}
trap cleanup EXIT TERM INT
ready=0
for ((attempt=0; attempt<180; attempt++)); do
    kill -0 "$comfy_pid" 2>/dev/null || { cat /tmp/comfyui.log; exit 1; }
    if curl --fail --silent http://127.0.0.1:8188/system_stats > /dev/null; then ready=1; break; fi
    sleep 2
done
if [ "$ready" != 1 ]; then cat /tmp/comfyui.log; echo 'ComfyUI startup timeout'; exit 1; fi
if ! python preflight.py; then cat /tmp/comfyui.log; exit 1; fi
python handler.py &
worker_pid=$!
wait -n "$worker_pid" "$comfy_pid"
