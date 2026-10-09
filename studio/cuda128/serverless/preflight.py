"""Validate both trusted graphs against the actual ComfyUI node registry at boot."""
import os
import requests
from graph import native, wrapper, validate_graph
from runtime_checks import check_onnx_cuda

check_onnx_cuda()

info = requests.get('http://127.0.0.1:8188/object_info', timeout=120).json()
params = dict(width=480, height=832, fps=30, steps=4, seed=0, prompt='A person moving naturally')
for profile in os.environ.get('WORKFLOW_PROFILES', 'animate-ki,animate-wrapper').split(','):
    builder = native if profile == 'animate-ki' else wrapper
    graph, _ = builder('preflight.png', 'preflight.mp4', params, 149, 'studio/preflight/result')
    # File names are job inputs, not installed model choices.
    for cls, field in [('LoadImage', 'image'), ('VHS_LoadVideo', 'video')]:
        if cls in info:
            spec = info[cls]['input']['required'][field]
            spec[0] = 'STRING'
    validate_graph(graph, info)
    print('Graph schema validated:', profile, len(graph), 'nodes', flush=True)
