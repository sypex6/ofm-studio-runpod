"""CPU build check: node imports and schemas, no weights or generation.

Temporary empty files exercise model filename discovery only and are removed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import requests
from graph import native, wrapper, validate_graph

root=Path(os.environ.get('COMFY_ROOT','/opt/ComfyUI'))
categories=['diffusion_models','vae','text_encoders','clip_vision','loras','controlnet','detection']
with tempfile.TemporaryDirectory(prefix='registry-check-') as temp:
    directory=Path(temp)
    for model in json.loads(Path(__file__).with_name('models.json').read_text()):
        p=directory/model['path'];p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    config=directory/'paths.yaml'
    config.write_text('test_models:\n  base_path: '+str(directory)+'\n'+''.join('  '+c+': '+c+'\n' for c in categories))
    process=subprocess.Popen([sys.executable,str(root/'main.py'),'--cpu','--listen','127.0.0.1','--port','8189',
                              '--disable-auto-launch','--extra-model-paths-config',str(config)],cwd=root)
    try:
        for _ in range(180):
            if process.poll() is not None:raise RuntimeError('ComfyUI CPU registry startup failed')
            try:
                response=requests.get('http://127.0.0.1:8189/object_info',timeout=5)
                response.raise_for_status();info=response.json();break
            except requests.RequestException:time.sleep(2)
        else:raise TimeoutError('Registry startup timed out')
        for cls,field in [('LoadImage','image'),('VHS_LoadVideo','video')]:
            info[cls]['input']['required'][field][0]='STRING'
        params=dict(width=480,height=832,fps=30,steps=4,seed=0,prompt='Natural movement')
        for builder in (native,wrapper):
            graph,_=builder('test.png','test.mp4',params,149,'studio/check/result')
            validate_graph(graph,info)
            print('Installed registry check passed:',builder.__name__,flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
