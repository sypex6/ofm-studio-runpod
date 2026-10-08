"""Build-time pinned node installation, preserving CUDA PyTorch versions."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ.get('COMFY_ROOT', '/opt/ComfyUI'))
lock = json.loads(Path(__file__).with_name('nodes.lock.json').read_text())
constraints = Path('/opt/torch-constraints.txt')
import torch, torchvision, torchaudio
constraints.write_text('\n'.join(f'{name}=={module.__version__}' for name, module in
                                [('torch', torch), ('torchvision', torchvision), ('torchaudio', torchaudio)]) + '\n')
for entry in lock:
    target = root / 'custom_nodes' / entry['name']
    subprocess.run(['git', 'clone', '--filter=blob:none', entry['url'], str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', entry['commit']], check=True)
    requirement = target / 'requirements.txt'
    if requirement.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-c', str(constraints),
                        '-r', str(requirement)], check=True)
