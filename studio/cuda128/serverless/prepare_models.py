"""Verified HTTP downloads onto Network or Global Volumes; no Hub/Xet cache."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlparse


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, model):
    return (path.is_file() and path.stat().st_size == model['size']
            and sha256(path) == model['sha256'])


def download(model, target, token=None, attempts=5, retry_delay=5, object_storage=False):
    import requests
    # On object storage write the final file directly. It remains unusable until
    # prepare() writes a verified completion marker. Never run workers during upload.
    temporary = target if object_storage else target.with_suffix(target.suffix + '.part')
    state = (target.with_suffix(target.suffix + '.download.json') if object_storage
             else temporary.with_suffix(temporary.suffix + '.json'))
    if object_storage:
        target.with_suffix(target.suffix + '.complete.json').unlink(missing_ok=True)
    try:
        previous = json.loads(state.read_text())
    except (OSError, ValueError):
        previous = None
    if previous != model['sha256']:
        temporary.unlink(missing_ok=True)
    state.write_text(json.dumps(model['sha256']))
    # Requests removes Authorization on redirects to a different host.
    headers = {'Accept-Encoding': 'identity'}
    if token and urlparse(model['url']).hostname == 'huggingface.co':
        headers['Authorization'] = 'Bearer ' + token
    for attempt in range(attempts):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == model['size'] and verified(temporary, model):
            if not object_storage:
                temporary.replace(target)
            state.unlink(missing_ok=True)
            return
        if offset >= model['size']:
            temporary.unlink()
            offset = 0
        if not object_storage:
            free = shutil.disk_usage(target.parent).free
            needed = model['size'] - offset
            if free < needed + 256 * 1024 * 1024:
                raise RuntimeError(f"Insufficient Volume space for {model['path']}: "
                                   f"free={free / 2**30:.2f} GiB, remaining={needed / 2**30:.2f} GiB. "
                                   'Increase Network Volume or inspect old cache/files.')
        request_headers = {**headers, **({'Range': f'bytes={offset}-'} if offset else {})}
        print(f"Downloading: {model['path']} from {offset / 2**30:.2f} GiB", flush=True)
        try:
            with requests.get(model['url'], headers=request_headers, stream=True,
                              timeout=(30, 120)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or int(match[3]) != model['size']:
                        raise RuntimeError('Invalid download range')
                elif response.status_code == 200:
                    offset = 0  # Range ignored: truncate this file instead of appending.
                else:
                    raise RuntimeError('Unexpected download status')
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise RuntimeError('Unexpected compressed download')
                with temporary.open('ab' if offset else 'wb') as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if stream.tell() + len(chunk) > model['size']:
                            raise RuntimeError('Download exceeds expected size')
                        stream.write(chunk)
                    stream.flush()
                    if not object_storage:
                        os.fsync(stream.fileno())
            if not verified(temporary, model):
                if temporary.stat().st_size == model['size']:
                    temporary.unlink()
                raise RuntimeError('Download size/SHA-256 mismatch')
            if not object_storage:
                temporary.replace(target)
            state.unlink(missing_ok=True)
            return
        except (requests.RequestException, RuntimeError) as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (401, 403, 404) or attempt + 1 == attempts:
                raise RuntimeError(f"Download failed: {model['path']} ({type(exc).__name__}, HTTP {status})") from None
        except OSError as exc:
            if exc.errno in (28, 122):
                raise RuntimeError('Storage quota exceeded; inspect volume configuration and account limits') from None
            if attempt + 1 == attempts:
                raise RuntimeError(f"Download failed: {model['path']} ({type(exc).__name__})") from None
        print(f"Retrying: {model['path']} ({attempt + 1}/{attempts})", flush=True)
        time.sleep(retry_delay)


def prepare(root, profiles, check=False, include_optional=False, readonly=False, global_write=False):
    if global_write and (readonly or check):
        raise ValueError('--global-write cannot be combined with --check or --readonly')
    if readonly and not check:
        raise ValueError('Global Volume is read-only in workers; publish models from a temporary Pod first')
    models = json.loads(Path(__file__).with_name('models.json').read_text())
    root = Path(root)
    if readonly:
        if not root.is_dir():
            raise SystemExit('Global Volume models directory is missing; publish models before starting workers')
    else:
        root.mkdir(parents=True, exist_ok=True)
    selected = [m for m in models if set(m['profiles']) & set(profiles)
                and (include_optional or not m.get('optional'))]
    print(f'Model Volume: {root}, readonly={readonly}; '
          f'selected models={sum(m["size"] for m in selected) / 2**30:.2f} GiB', flush=True)
    if global_write:
        print('Global Volume direct upload: single writer only; start workers AFTER completion.', flush=True)
    if not readonly and not global_write:
        usage = shutil.disk_usage(root)
        print(f'Volume total={usage.total / 2**30:.2f} GiB, free={usage.free / 2**30:.2f} GiB', flush=True)
    missing = []
    if readonly or global_write:
        guard = nullcontext()
    else:
        from filelock import FileLock
        guard = FileLock(str(root / '.prepare.lock'), timeout=7200)
    with guard:
        for model in selected:
            target = root / model['path']
            if not target.resolve().is_relative_to(root.resolve()):
                raise RuntimeError('Model path leaves the Volume')
            marker = target.with_suffix(target.suffix + '.complete.json')
            fingerprint = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
            try:
                previous = json.loads(marker.read_text())
            except (OSError, ValueError):
                previous = {}
            if (target.is_file() and target.stat().st_size == model['size']
                    and previous.get('fingerprint') == fingerprint
                    and previous.get('sha256') == model['sha256']):
                continue
            if check:
                missing.append(model['path'])
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Invalidate readiness before replacing any bytes, even if a prior
            # upload used another manifest. Global Volume has no atomic rename.
            if global_write:
                marker.unlink(missing_ok=True)
            if not verified(target, model):
                # Adopt old Hub blobs without creating another full copy.
                candidates = ([] if global_write else
                              list((root / '.hf-cache').glob(f"*/blobs/{model['sha256']}")))
                for cached in candidates:
                    if verified(cached, model):
                        temporary = target.with_suffix(target.suffix + '.part')
                        temporary.unlink(missing_ok=True)
                        os.link(cached, temporary)
                        temporary.replace(target)
                        break
                else:
                    download(model, target, os.environ.get('HF_TOKEN') or None,
                             object_storage=global_write)
            marker_temp = marker if global_write else marker.with_suffix(marker.suffix + '.part')
            marker_temp.write_text(json.dumps({'size': target.stat().st_size,
                                              'fingerprint': fingerprint, 'sha256': model['sha256']}))
            if not global_write:
                marker_temp.replace(marker)
            print('Ready:', model['path'], flush=True)
    if missing:
        hint = ('Publish models from a temporary Pod, then replace workers.' if readonly
                else 'Set DOWNLOAD_MODELS=1 first.')
        raise SystemExit('Missing/incomplete models: ' + ', '.join(missing) + '. ' + hint)
    print('Models ready: ' + str(len(selected)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/runpod-volume/models')
    parser.add_argument('--profiles', default='animate-ki,animate-wrapper')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--include-optional', action='store_true')
    parser.add_argument('--readonly', action='store_true', help='Check Global Volume without locks or writes')
    parser.add_argument('--global-write', action='store_true',
                        help='Single-writer direct upload on a temporary Pod; no staging, locks or rename')
    args = parser.parse_args()
    profiles = args.profiles.split(',')
    if not set(profiles) <= {'animate-ki', 'animate-wrapper'}:
        parser.error('Unknown profile')
    prepare(args.root, profiles, args.check, args.include_optional, args.readonly, args.global_write)
