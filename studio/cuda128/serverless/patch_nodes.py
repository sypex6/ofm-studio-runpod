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
