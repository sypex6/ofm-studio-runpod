"""Runpod worker: download two inputs, execute trusted graph, PUT MP4 to Studio."""
import ipaddress
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse
import uuid

import requests

try:
    from .graph import native, wrapper, validate_graph
    from .events import ComfyEvents
except ImportError:
    from graph import native, wrapper, validate_graph
    from events import ComfyEvents

COMFY = 'http://127.0.0.1:8188'
ROOT = Path(os.environ.get('COMFY_ROOT', '/opt/ComfyUI'))
LOCK = threading.Lock()  # One generation per GPU, also protects ComfyUI interrupt.
TIMEOUT = int(os.environ.get('GENERATION_TIMEOUT', '3600'))


def public_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError('Expected a public HTTPS URL')
    for item in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(item[4][0]).is_global:
            raise ValueError('Private network URLs are prohibited')
    return url


def fetch(url, target, limit):
    """Check each redirect separately; never send API tokens to input URLs."""
    for _ in range(5):
        public_url(url)
        with requests.get(url, stream=True, allow_redirects=False, timeout=(20, 120)) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                from urllib.parse import urljoin
                url = urljoin(url, response.headers['Location'])
                continue
            response.raise_for_status()
            total = 0
            with target.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise ValueError('Input file exceeds its size limit')
                    stream.write(chunk)
            if not total:
                raise ValueError('Empty input')
            return
    raise ValueError('Too many redirects')


def checked_input(data):
    if not isinstance(data, dict):
        raise ValueError('input must be an object')
    workflow = data.get('workflow')
    if workflow not in ('animate-ki', 'animate-wrapper'):
        raise ValueError('Unknown workflow')
    if workflow not in os.environ.get('WORKFLOW_PROFILES', 'animate-ki,animate-wrapper').split(','):
        raise ValueError('This workflow profile is not enabled on the endpoint')
    if not isinstance(data.get('parameters', {}), dict):
        raise ValueError('parameters must be an object')
    params = {'width': 480, 'height': 832, 'fps': 30, 'steps': 4, 'seed': 0, 'prompt': '', 'duration': 5, **data.get('parameters', {})}
    for name, minimum, maximum in [('width', 256, 1280), ('height', 256, 1280), ('fps', 8, 30), ('steps', 1, 30), ('seed', 0, 2**53-1), ('duration', 1, 30)]:
        value = params[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f'Invalid {name}')
    if params['width'] % 16 or params['height'] % 16 or params['width'] * params['height'] > 1280 * 720:
        raise ValueError('Dimensions must be multiples of 16, at most 921600 pixels')
    if not isinstance(params['prompt'], str) or len(params['prompt']) > 32000:
        raise ValueError('Invalid prompt')
    upload = public_url(data.get('result_upload_url', ''))
    # Limit delivery to the Studio hostname configured by the endpoint owner.
    host = os.environ.get('STUDIO_HOST', '').lower()
    if not host or urlparse(upload).hostname.lower() != host:
        raise ValueError('result_upload_url must belong to STUDIO_HOST')
    return workflow, params, upload


def execute(graph, output_node, job, progress):
    print('ComfyUI: validating workflow', flush=True)
    registry = requests.get(COMFY + '/object_info', timeout=60)
    registry.raise_for_status()
    info = registry.json()
    validate_graph(graph, info)
    print('ComfyUI: submitting workflow', flush=True)
    client_id=uuid.uuid4().hex
    events=ComfyEvents(COMFY,client_id)
    try:
        response = requests.post(COMFY + '/prompt', json={'prompt': graph, 'client_id': client_id}, timeout=60)
        if response.status_code != 200:
            detail = response.json().get('node_errors', {})
            raise RuntimeError('ComfyUI rejected workflow: ' + json.dumps(detail)[:2000])
        prompt_id = response.json()['prompt_id']
    except BaseException:
        events.close()
        raise
    print('ComfyUI: accepted prompt ' + prompt_id, flush=True)
    started = time.monotonic()
    heartbeat = started + 30
    deadline = time.monotonic() + TIMEOUT
    last_node=None
    try:
        while time.monotonic() < deadline:
            history = requests.get(COMFY + '/history/' + prompt_id, timeout=30)
            history.raise_for_status()
            result = history.json().get(prompt_id)
            observed=events.state(prompt_id)
            if observed.get('error'):
                raise RuntimeError('ComfyUI: '+observed['error'])
            node=observed.get('node')
            if node and node!=last_node:
                last_node=node
                label=graph.get(node,{}).get('class_type',node)
                message='ComfyUI: node '+node+' '+label
                print(message,flush=True)
                progress(job,message)
            final_event = observed.get('outputs',{}).get(output_node,{})
            # In this trusted graph the selected output node is the terminal
            # video encoder. Its executed event means the encoder returned and
            # closed the output, even if aggregate history has not arrived yet.
            # A file alone, preview node or unfinished batched encoder is insufficient.
            final_finished = bool(final_event) and not final_event.get('unfinished_batch')
            if result or final_finished:
                status = (result or {}).get('status', {})
                if status.get('status_str') == 'error':
                    raise RuntimeError('ComfyUI generation failed; see worker logs')
                if status.get('completed') or final_finished:
                    outputs = (result or {}).get('outputs', {}).get(output_node, {}) or final_event
                    for video in reversed(outputs.get('gifs', []) + outputs.get('videos', [])):
                        directory = ROOT / ('output' if video.get('type') == 'output' else 'temp')
                        path = (directory / video.get('subfolder', '') / video['filename']).resolve()
                        if path.is_relative_to(directory.resolve()) and path.suffix.lower() == '.mp4' and path.is_file():
                            return path
                    raise RuntimeError('Workflow completed without an MP4 output')
            if time.monotonic() >= heartbeat:
                label=graph.get(last_node,{}).get('class_type','awaiting execution')
                step=observed.get('progress')
                message=f'ComfyUI: {label}, elapsed={int(time.monotonic()-started)}s'+(f', step={step[0]}/{step[1]}' if step else '')
                print(message,flush=True)
                progress(job,message)
                heartbeat = time.monotonic() + 30
            time.sleep(2)
        raise TimeoutError('ComfyUI generation timeout')
    except BaseException:
        try:requests.post(COMFY + '/interrupt', timeout=10)
        except requests.RequestException:print('ComfyUI interrupt request failed',flush=True)
        raise
    finally:
        events.close()
        try:requests.post(COMFY + '/history', json={'delete': [prompt_id]}, timeout=10)
        except requests.RequestException:print('ComfyUI history cleanup request failed',flush=True)


def handler(job, progress=None):
    if progress is None:
        import runpod
        progress = runpod.serverless.progress_update
    with LOCK:
        def report(message):
            print('Studio worker: ' + message, flush=True)
            progress(job, message)
        workflow, params, upload = checked_input(job['input'])
        token = uuid.uuid4().hex
        # LoadImage's combo enumerates only files directly inside input/.
        # UUID filenames keep jobs isolated while remaining discoverable by
        # ComfyUI's actual schema and prompt validation.
        directory = ROOT / 'input'
        directory.mkdir(parents=True, exist_ok=True)
        image, video = directory / (token + '_reference.png'), directory / (token + '_motion.mp4')
        prefix = 'studio/' + token + '/result'
        try:
            report('Downloading inputs')
            fetch(job['input']['image_url'], image, 20 * 1024**2)
            fetch(job['input']['video_url'], video, 512 * 1024**2)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                    'stream=width,height,duration:format=duration', '-of', 'json', str(video)],
                                   check=True, capture_output=True, text=True, timeout=30)
            metadata = json.loads(probe.stdout)
            streams = metadata.get('streams', [])
            if not streams or streams[0]['width'] * streams[0]['height'] > 4096 * 4096:
                raise ValueError('Unsupported reference video dimensions')
            duration = float(metadata.get('format', {}).get('duration') or streams[0].get('duration', 0))
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('Reference video has no duration')
            requested = min(params['duration'], duration)
            frames = max(1, int(requested * params['fps']))
            # Wan requires 4n+1 frames; use only frames actually present in reference.
            frames = max(1, ((frames - 1) // 4) * 4 + 1)
            report('Generating video')
            builder = native if workflow == 'animate-ki' else wrapper
            graph, output = builder(image.name, video.name, params, frames, prefix)
            path = execute(graph, output, job, progress)
            if path.stat().st_size > 512 * 1024**2:
                raise ValueError('Result exceeds Studio video size limit')
            report('Saving result to Studio')
            # Retrying delivery does not repeat GPU generation; callback is idempotent.
            for attempt in range(3):
                try:
                    with path.open('rb') as stream:
                        response = requests.put(upload, data=stream, headers={'Content-Type': 'video/mp4'},
                                                timeout=(20, 600), allow_redirects=False)
                    response.raise_for_status()
                    url = response.json()['url']
                    public_url(url)
                    return {'outputs': [url], 'frames': frames, 'fps': params['fps'], 'workflow': workflow}
                except requests.RequestException:
                    if attempt == 2:
                        raise RuntimeError('Result delivery failed; see worker logs') from None
                    time.sleep(3)
        finally:
            image.unlink(missing_ok=True)
            video.unlink(missing_ok=True)
            for kind in ('output', 'temp'):
                shutil.rmtree(ROOT / kind / 'studio' / token, ignore_errors=True)
            # Drop cached input/output tensors so a warm worker does not retain user data.
            try:
                requests.post(COMFY + '/free', json={'unload_models': False, 'free_memory': True}, timeout=30)
            except requests.RequestException:
                print('ComfyUI cache cleanup request failed; result delivery status preserved',flush=True)


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
