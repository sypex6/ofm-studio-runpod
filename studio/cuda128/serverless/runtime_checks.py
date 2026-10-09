"""Fail startup when ONNX CUDA libraries are incompatible, instead of CPU fallback."""


def small_onnx_model():
    # Minimal standard ONNX Add graph (IR 8, opset 13), encoded without needing
    # the optional onnx compiler package in the worker image.
    def varint(number):
        value = bytearray()
        while number > 127:
            value.append((number & 127) | 128)
            number >>= 7
        value.append(number)
        return bytes(value)
    def field(number, value):
        if isinstance(value, int):
            return varint(number << 3) + varint(value)
        return varint((number << 3) | 2) + varint(len(value)) + value
    shape = field(1, field(1, 1))
    tensor_type = field(1, 1) + field(2, shape)
    def value_info(name):
        return field(1, name) + field(2, field(1, tensor_type))
    node = field(1, b'x') + field(1, b'x') + field(2, b'y') + field(4, b'Add')
    graph = field(1, node) + field(2, b'cuda-preflight') + field(11, value_info(b'x')) + field(12, value_info(b'y'))
    return field(1, 8) + field(7, graph) + field(8, field(2, 13))


def check_onnx_cuda():
    import torch  # Preload the matching PyTorch CUDA/cuDNN libraries first.
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(small_onnx_model(), providers=['CUDAExecutionProvider'])
    if 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('ONNX CUDA provider failed to initialize; check CUDA/cuDNN libraries. CPU fallback refused.')
    session.disable_fallback()
    result = session.run(None, {'x': np.array([1], dtype=np.float32)})[0]
    if not np.array_equal(result, np.array([2], dtype=np.float32)):
        raise RuntimeError('ONNX CUDA smoke test returned incorrect output')
    print('ONNX CUDA preflight passed: ' + ort.__version__, flush=True)
