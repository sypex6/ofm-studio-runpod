# Build context: studio/cuda128. Models live on an attached Volume, never in the image.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    COMFY_ROOT=/opt/ComfyUI HF_HOME=/tmp/huggingface HF_HUB_DISABLE_XET=1 \
    WORKFLOW_PROFILES=animate-ki,animate-wrapper
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG COMFY_COMMIT=b00c6e95279053474955540ba4f551646722b9aa
RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout ${COMFY_COMMIT}
WORKDIR /opt/ComfyUI
RUN python -c "import torch,torchvision,torchaudio; open('/opt/torch-constraints.txt','w').write('\n'.join(n+'=='+m.__version__ for n,m in [('torch',torch),('torchvision',torchvision),('torchaudio',torchaudio)])+'\n')" \
    && pip install --no-cache-dir -c /opt/torch-constraints.txt -r requirements.txt \
    && pip install --no-cache-dir -c /opt/torch-constraints.txt runpod requests filelock huggingface_hub onnxruntime-gpu color-matcher
COPY serverless /opt/studio-worker/serverless
RUN python /opt/studio-worker/serverless/install_nodes.py \
    && python /opt/studio-worker/serverless/patch_nodes.py \
    && python -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda"
COPY workflows /opt/studio-worker/workflows
RUN python /opt/studio-worker/serverless/check_registry.py
COPY start.sh /opt/studio-worker/start.sh
RUN sed -i 's/\r$//' /opt/studio-worker/start.sh && chmod +x /opt/studio-worker/start.sh
WORKDIR /opt/studio-worker/serverless
ENTRYPOINT ["/bin/bash", "/opt/studio-worker/start.sh"]
