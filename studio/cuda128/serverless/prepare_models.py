"""Download models once onto the network volume, with atomic promotion and locks."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def prepare(root, profiles, check=False):
    from filelock import FileLock
    from huggingface_hub import hf_hub_download
    models = json.loads(Path(__file__).with_name('models.json').read_text())
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    missing = []
    with FileLock(str(root / '.prepare.lock'), timeout=7200):
        for model in models:
            if not set(model['profiles']) & set(profiles):
                continue
            target = root / model['path']
            marker = target.with_suffix(target.suffix + '.complete.json')
            fingerprint = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
            if target.is_file() and target.stat().st_size > 0 and marker.exists():
                previous = json.loads(marker.read_text())
                if previous.get('fingerprint') == fingerprint and previous.get('size') == target.stat().st_size:
                    continue
            if check:
                missing.append(model['path'])
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            cached = hf_hub_download(model['repo'], model['file'], revision=model.get('revision', 'main'),
                                     token=os.environ.get('HF_TOKEN') or None,
                                     cache_dir=root / '.hf-cache')
            if model.get('size') and Path(cached).stat().st_size != model['size']:
                raise RuntimeError('Unexpected model size: ' + model['path'])
            temporary = target.with_suffix(target.suffix + '.part')
            # Keep just one full copy; hub cache and models are on the same volume.
            temporary.unlink(missing_ok=True)
            os.link(Path(cached).resolve(), temporary)
            temporary.replace(target)
            marker.write_text(json.dumps({'size': target.stat().st_size, 'fingerprint': fingerprint}))
            print('Ready:', model['path'], flush=True)
    if missing:
        raise SystemExit('Missing/incomplete models: ' + ', '.join(missing) + '. Run prepare_models.py before enabling workers.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/runpod-volume/models')
    parser.add_argument('--profiles', default='animate-ki,animate-wrapper')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    profiles = args.profiles.split(',')
    if not set(profiles) <= {'animate-ki', 'animate-wrapper'}:
        parser.error('Unknown profile')
    prepare(args.root, profiles, args.check)
