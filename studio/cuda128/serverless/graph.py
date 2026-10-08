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
    """Kiara Animate: unroll its loop into overlapping 77-frame blocks."""
    original = json.loads((WORKFLOWS / 'Kiara Animate API.json').read_text(encoding='utf-8'))
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
    loader = copy.deepcopy(original['356']['inputs'])
    # Same basename as SSIBAL, but the supplied repositories contain different weights.
    loader['unet_name'] = 'kiara/' + loader['unet_name']
    model = g.add('UNETLoader', **loader)
    for ident in (355, 327):
        inputs = copy.deepcopy(original[str(ident)]['inputs'])
        inputs['model'] = model
        model = g.add('LoraLoaderModelOnly', **inputs)
    # Use SDPA and offloading; CUDA compilation and SageAttention are optional optimizations,
    # not dependencies of the animation or cold-start requirements.
    model = g.add('ModelSamplingSD3', model=model, shift=original['330']['inputs']['shift'])
    clip = g.add('CLIPLoader', **original['333']['inputs'])
    vae = g.add('VAELoader', **original['329']['inputs'])
    vision = g.add('CLIPVisionLoader', **original['348']['inputs'])
    vision = g.add('CLIPVisionEncode', clip_vision=vision, image=ref, crop='none')
    pos = g.add('CLIPTextEncode', clip=clip, text=params['prompt'])
    neg = g.add('CLIPTextEncode', clip=clip, text=original['335']['inputs']['text'])
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
    """Use the supplied SSIBAL API export; retain only final-output dependencies."""
    nodes = json.loads((WORKFLOWS / 'SSIBAL Animate API.json').read_text(encoding='utf-8'))
    graph = {}
    def resolve(node_id, slot=0):
        node = nodes[str(node_id)]
        if node['class_type'] == 'easy float':
            return node['inputs']['value']
        key = str(node_id)
        if key in graph:
            return [key, slot]
        graph[key] = {'class_type': node['class_type'], 'inputs': {}}
        inputs = copy.deepcopy(node['inputs'])
        for name, value in inputs.items():
            if name in ('width', 'height'):
                inputs[name] = params[name]
            elif isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
                inputs[name] = resolve(*value)
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
            if inputs.get('batched_cfg') == '':
                inputs['batched_cfg'] = False
        elif key == '354':
            inputs['lora_1'] = 'Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1-low_noise_model.safetensors'
        elif key == '270':
            inputs.update(frame_window_size=81, num_frames=frames)
        elif key == '562':
            inputs['chunk_size'] = 81
        elif key == '751':
            inputs.update(frame_rate=params['fps'], filename_prefix=prefix, save_output=True)
        return [key, slot]
    resolve('751')
    graph['studio_frame_limit'] = {'class_type': 'GetImageRangeFromBatch',
                                  'inputs': {'images': ['562', 0], 'start_index': 0, 'num_frames': frames}}
    graph['751']['inputs']['images'] = ['studio_frame_limit', 0]
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
