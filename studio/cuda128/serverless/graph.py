"""Trusted workflow builders. Clients cannot submit arbitrary ComfyUI graphs."""
import copy
import json
import math
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / 'workflows'


class Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, cls, slot=0, **inputs):
        key = str(len(self.nodes) + 1)
        self.nodes[key] = {'class_type': cls, 'inputs': inputs}
        return [key, slot]


def combine_inputs(images, audio, fps, prefix):
    return dict(images=images, audio=audio, frame_rate=fps, loop_count=0,
                filename_prefix=prefix, format='video/h264-mp4', pix_fmt='yuv420p',
                crf=19, save_metadata=False, trim_to_audio=False, pingpong=False, save_output=True)


def native(image, video, params, frames, prefix):
    """Animate X: unroll the UI loop into explicit overlapping 77-frame blocks."""
    source = json.loads(next(WORKFLOWS.glob('Animate X*.json')).read_text(encoding='utf-8'))
    original = {n['id']: n for n in source['nodes']}
    g = Graph()
    w, h, fps = params['width'], params['height'], params['fps']
    ref = g.add('LoadImage', image=image)
    ref = g.add('ImageResizeKJv2', image=ref, width=w, height=h, upscale_method='lanczos',
                keep_proportion='crop', pad_color='0, 0, 0', crop_position='center', divisible_by=16, device='cpu')
    vid = g.add('VHS_LoadVideo', video=video, force_rate=fps, custom_width=w, custom_height=h,
                frame_load_cap=frames, skip_first_frames=0, select_every_nth=1, format='Wan')
    detector = g.add('OnnxDetectionModelLoader', vitpose_model='vitpose-l-wholebody.onnx',
                     yolo_model='yolov10m.onnx', onnx_device='CUDAExecutionProvider')
    faces = g.add('PoseAndFaceDetection', slot=1, model=detector, images=vid, width=w, height=h)
    poses = g.add('DrawViTPose', pose_data=[faces[0], 0], width=w, height=h,
                 retarget_padding=10, body_stick_width=-1, hand_stick_width=-1, draw_head='True')
    model = g.add('UNETLoader', unet_name='WanModel.safetensors', weight_dtype='fp8_e4m3fn_fast')
    for ident in (355, 327):
        name, strength = original[ident]['widgets_values']
        model = g.add('LoraLoaderModelOnly', model=model, lora_name=name, strength_model=strength)
    # Use SDPA and offloading; CUDA compilation and SageAttention are optional optimizations,
    # not dependencies of the animation or cold-start requirements.
    model = g.add('ModelSamplingSD3', model=model, shift=8.0)
    clip = g.add('CLIPLoader', clip_name='text_enc.safetensors', type='wan', device='default')
    vae = g.add('VAELoader', vae_name='vae.safetensors')
    vision = g.add('CLIPVisionLoader', clip_name='klip_vision.safetensors')
    vision = g.add('CLIPVisionEncode', clip_vision=vision, image=ref, crop='none')
    pos = g.add('CLIPTextEncode', clip=clip, text=params['prompt'])
    neg = g.add('CLIPTextEncode', clip=clip, text=original[335]['widgets_values'][0])
    previous, accumulated, offset, produced = None, None, 0, 0
    while produced < frames:
        overlap = 5 if previous else 0
        length = min(77, math.ceil((frames - produced + overlap - 1) / 4) * 4 + 1)
        inputs = dict(positive=pos, negative=neg, vae=vae, width=w, height=h,
                      length=length, batch_size=1, continue_motion_max_frames=5,
                      video_frame_offset=offset, reference_image=ref, clip_vision_output=vision,
                      face_video=faces, pose_video=poses)
        if previous:
            inputs['continue_motion'] = previous
        conditioning = g.add('WanAnimateToVideo', **inputs)
        sampled = g.add('KSampler', model=model, positive=conditioning, negative=[conditioning[0], 1],
                        latent_image=[conditioning[0], 2], seed=params['seed'], steps=params['steps'],
                        cfg=1.0, sampler_name='sa_solver', scheduler='linear_quadratic', denoise=1.0)
        trimmed = g.add('TrimVideoLatent', samples=sampled, trim_amount=[conditioning[0], 3])
        decoded = g.add('VAEDecode', samples=trimmed, vae=vae)
        fresh = g.add('GetImageRangeFromBatch', images=decoded,
                      start_index=[conditioning[0], 4], num_frames=min(length - overlap, frames - produced))
        accumulated = g.add('ImageBatch', image1=accumulated, image2=fresh) if accumulated else fresh
        previous, offset = decoded, [conditioning[0], 5]
        produced += length - overlap
    output = g.add('VHS_VideoCombine', **combine_inputs(accumulated, [vid[0], 2], fps, prefix))
    return g.nodes, output[0]


def wrapper(image, video, params, frames, prefix):
    """Resolve frontend Set/Get variables; preserve the original backend graph."""
    doc = json.loads(next(WORKFLOWS.glob('SSIBALHUB*.json')).read_text(encoding='utf-8'))
    nodes = {str(n['id']): n for n in doc['nodes']}
    links = {l[0]: [str(l[1]), l[2]] for l in doc['links']}
    setters = {n['widgets_values'][0]: n for n in nodes.values() if n['type'] == 'SetNode'}
    graph = {}
    def resolve(node_id, slot=0):
        node = nodes[str(node_id)]
        if node['type'] in ('GetNode', 'SetNode'):
            name = node['widgets_values'][0]
            if name in ('width', 'height', 'num_frames'):
                return {'width': params['width'], 'height': params['height'], 'num_frames': frames}[name]
            parent = setters[name]
            return resolve(*links[parent['inputs'][0]['link']])
        if node['type'] == 'easy float':
            return node['widgets_values'][0]
        key = str(node_id)
        if key in graph:
            return [key, slot]
        graph[key] = {'class_type': node['type'], 'inputs': {}}
        values = copy.deepcopy(node.get('widgets_values') or [])
        widgets = values if isinstance(values, dict) else {}
        if isinstance(values, list):
            cursor = 0
            for inp in node.get('inputs', []):
                if inp.get('widget'):
                    widgets[inp['name']] = values[cursor]
                    cursor += 1
                    if inp['name'] in ('seed', 'noise_seed') and cursor < len(values) and values[cursor] in ('randomize', 'fixed', 'increment', 'decrement'):
                        cursor += 1
        inputs = {k: v for k, v in widgets.items() if k != 'videopreview'}
        for inp in node.get('inputs', []):
            if inp.get('link') is not None:
                inputs[inp['name']] = resolve(*links[inp['link']])
        graph[key]['inputs'] = inputs
        if key == '76':
            inputs['image'] = image
        elif key == '75':
            inputs.update(video=video, force_rate=params['fps'], custom_width=params['width'],
                          custom_height=params['height'], frame_load_cap=frames, skip_first_frames=0, select_every_nth=1)
        elif key == '493':
            inputs['text'] = params['prompt']
        elif key == '273':
            inputs.update(seed=params['seed'], steps=params['steps'])
        elif key == '354':
            inputs['lora_1'] = 'Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1-low_noise_model.safetensors'
        elif key == '270':
            inputs['frame_window_size'] = 81
        elif key == '562':
            # TS node otherwise writes diagnostic copies of every window.
            inputs.update(fps=params['fps'], save_individual_chunks=False, debug=False)
        elif key == '751':
            inputs.update(frame_rate=params['fps'], filename_prefix=prefix, save_output=True)
        return [key, slot]
    resolve('751')
    return graph, '751'


def validate_graph(graph, object_info):
    """Fail before generation if a pinned node/schema/model is unavailable."""
    for key, node in graph.items():
        cls, inputs = node['class_type'], node['inputs']
        if cls not in object_info:
            raise ValueError(f'Missing ComfyUI node: {cls}')
        info = object_info[cls]['input']
        specs = {**info.get('required', {}), **info.get('optional', {})}
        # Frontend output-preview and widget-only decorations are not API inputs.
        node['inputs'] = inputs = {k: v for k, v in inputs.items() if k in specs}
        for name, spec in info.get('required', {}).items():
            if name not in inputs:
                options = spec[1] if len(spec) > 1 else {}
                if 'default' in options:
                    inputs[name] = options['default']
                else:
                    raise ValueError(f'Missing required input: {cls}.{name}')
        for name, value in inputs.items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
                if value[0] not in graph:
                    raise ValueError(f'Broken graph link: {cls}.{name}')
            elif isinstance(specs[name][0], list) and value not in specs[name][0]:
                raise ValueError(f'Unavailable model/option: {cls}.{name} = {value}')
    return graph
