"""Single-writer model upload to object-backed storage; no locks, renames or links.

Run on a temporary Pod before attaching the volume to inference workers.
Source models are a Network Volume or an ordinary local staging directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

try:
    from .prepare_models import verified
except ImportError:
    from prepare_models import verified


def publish(source, destination, profiles, models=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Source and destination must be separate directories')
    if models is None:
        models = json.loads(Path(__file__).with_name('models.json').read_text())
    selected = [m for m in models if set(m['profiles']) & set(profiles) and not m.get('optional')]
    # Validate every source before changing anything on the destination.
    for model in selected:
        origin = source / model['path']
        target = destination / model['path']
        if not origin.resolve().is_relative_to(source) or not target.resolve().is_relative_to(destination):
            raise ValueError('Model path leaves storage directory')
        if not verified(origin, model):
            raise RuntimeError('Source missing or SHA-256 mismatch: ' + model['path'])
    for model in selected:
        origin, target = source / model['path'], destination / model['path']
        marker = target.with_suffix(target.suffix + '.complete.json')
        fingerprint = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        metadata = {'size': model['size'], 'fingerprint': fingerprint, 'sha256': model['sha256']}
        target.parent.mkdir(parents=True, exist_ok=True)
        # An interrupted upload has no completion marker. Repeat uploads recover it.
        marker.unlink(missing_ok=True)
        if not verified(target, model):
            print('Publishing: ' + model['path'], flush=True)
            shutil.copyfile(origin, target)
            if not verified(target, model):
                raise RuntimeError('Uploaded file SHA-256 mismatch: ' + model['path'])
        # Direct write: Global Volumes do not support atomic rename.
        marker.write_text(json.dumps(metadata), encoding='utf-8')
        print('Published: ' + model['path'], flush=True)
    print('Global Volume ready: ' + str(len(selected)) + ' models. Start NEW workers now.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--profiles', default='animate-ki,animate-wrapper')
    args = parser.parse_args()
    profiles = args.profiles.split(',')
    if not set(profiles) <= {'animate-ki', 'animate-wrapper'}:
        parser.error('Unknown profile')
    publish(args.source, args.destination, profiles)
