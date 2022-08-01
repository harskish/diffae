# Use conda env tvm (py38)

import numpy as np
import onnxruntime as ort # pip install onnxruntime
from bench import CONFIGS, show
from tqdm import trange
#import tvm # from source, v0.9.0

# Export
#_ = CONFIGS['cpu_traced']('ffhq256', export=True)
lat_init = 'ffhq256_lat_init.onnx'
lat_step = 'ffhq256_lat.onnx'
img_init = 'ffhq256_img_init.onnx'
img_step = 'ffhq256_img.onnx'

# In order of preference
backends = ['CPUExecutionProvider', 'CUDAExecutionProvider']

# Run
T = np.array([10], dtype=np.int64)
x = np.random.randn(1, 512).astype(np.float32)

sess = ort.InferenceSession(lat_init, providers=backends)
lat_params = sess.run(input_feed={ 'T': T },
    output_names=[
        'timestep_map',
        'posterior_variance',
        'alphas_cumprod',
        'alphas_cumprod_prev',
        'sqrt_recip_alphas_cumprod',
        'sqrt_recipm1_alphas_cumprod',
        'betas',
    ])

sess = ort.InferenceSession(lat_step, providers=backends)
for i in range(T.item()):
    x_out = sess.run(input_feed={
        't': (T - i - 1),
        'x': x,
        'timestep_map': lat_params[0].astype(np.int64),
        #'posterior_variance': lat_params[1], # optimized away?
        'alphas_cumprod': lat_params[2],
        'alphas_cumprod_prev': lat_params[3],
        'sqrt_recip_alphas_cumprod': lat_params[4],
        'sqrt_recipm1_alphas_cumprod': lat_params[5],
        #'betas': lat_params[6], # optimized away?
    }, output_names=['lats'])
    x = x_out[0]

lats = x

sess = ort.InferenceSession(img_init, providers=backends)
img_params = sess.run(input_feed={ 'T': T },
    output_names=[
        'timestep_map',
        'posterior_variance',
        'alphas_cumprod',
        'alphas_cumprod_prev',
        'sqrt_recip_alphas_cumprod',
        'sqrt_recipm1_alphas_cumprod',
        'betas',
    ])

x = np.random.randn(1, 3, 256, 256).astype(np.float32)
sess = ort.InferenceSession(img_step, providers=backends)
for i in trange(T.item()):
    x_out = sess.run(input_feed={
        't': (T - i - 1),
        'x': x,
        'lats': lats,
        'timestep_map': lat_params[0].astype(np.int64),
        #'posterior_variance': lat_params[1], # optimized away?
        'alphas_cumprod': lat_params[2],
        'alphas_cumprod_prev': lat_params[3],
        'sqrt_recip_alphas_cumprod': lat_params[4],
        'sqrt_recipm1_alphas_cumprod': lat_params[5],
        #'betas': lat_params[6], # optimized away?
    }, output_names=['output'])
    x = x_out[0]
    show(x)

print('Done')
