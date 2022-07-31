# Use conda env tvm (py38)

import numpy as np
import onnxruntime as ort # pip install onnxruntime
#import tvm # from source, v0.9.0

# Device used to export might not matter?
pth_lat = 'cuda_ffhq256_lat.onnx'
pth_img = 'cuda_ffhq256_img.onnx'

ort_session = ort.InferenceSession(pth_lat, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
outputs = ort_session.run(
    output_names=None,
    input_feed={
        'z': np.random.randn(1, 512).astype(np.float32),
        't': np.array([9], dtype=np.int64),
    },
)
print(outputs[0])


