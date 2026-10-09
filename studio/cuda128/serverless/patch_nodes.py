"""Small checked patch to the pinned TS node: preserve colors, omit extra videos."""
import os
from pathlib import Path

root = Path(os.environ.get('COMFY_ROOT', '/opt/ComfyUI'))
path = root / 'custom_nodes/comfyui-teskors-utils/nodes/color_match.py'
source = path.read_text(encoding='utf-8')
changes = {
    'SAVE_TEMP_CHUNKS = True': 'SAVE_TEMP_CHUNKS = False',
    'out_final = self._ensure_mp4v_writer(final_path_tmp, fps, w, h)': 'out_final = _DiscardVideoWriter()',
    'shutil.move(final_path_tmp, final_path)': '# Final MP4 is saved by TSVideoCombineNoMetadata.',
}
for before, after in changes.items():
    if source.count(before) != 1:
        raise RuntimeError('Pinned TSColorMatch source changed: cannot apply storage patch')
    source = source.replace(before, after)
source += '\n\nclass _DiscardVideoWriter:\n    def write(self, frame):\n        pass\n\n    def release(self):\n        pass\n'
compile(source, str(path), 'exec')
path.write_text(source, encoding='utf-8')
print('TSColorMatch: color processing preserved; extra MP4/chunk files disabled', flush=True)

# WAN's unmerged LoRAs are copied to CUDA later by the sampler. If CPU tensors
# still map remote object storage, every copy can fault into a slow remote read.
# Materialize only LoRAs, preserving all weights, strengths and precision.
wan = root / 'custom_nodes/ComfyUI-WanVideoWrapper/nodes_model_loading.py'
source = wan.read_text(encoding='utf-8')
before = 'load_torch_file(lora_path, safe_load=True)'
if source.count(before) != 2:
    raise RuntimeError('Pinned WAN LoRA loader changed: cannot apply eager-read patch')
source = source.replace(before, '_studio_load_lora(lora_path)')
source += '''

def _studio_load_lora(path):
    from pathlib import Path
    from safetensors.torch import load
    if Path(path).suffix.lower() != '.safetensors':
        return load_torch_file(path, safe_load=True)
    log.info('Studio: reading LoRA into RAM: %s', Path(path).name)
    tensors = load(Path(path).read_bytes())
    log.info('Studio: LoRA RAM read complete: %s', Path(path).name)
    return tensors
'''
compile(source, str(wan), 'exec')
wan.write_text(source, encoding='utf-8')
print('WAN: eager LoRA reads enabled; no network-backed LoRA tensors', flush=True)
