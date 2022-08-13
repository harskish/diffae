# Use conda env tvm (py38)

from multiprocessing.pool import RUN
from random import Random
from sympy import comp
import time
import torch
import numpy as np
from numpy.random import RandomState
import onnxruntime as ort # pip install onnxruntime
#import onnx # from src, whl/onnx. Also: conda install protobuf
from bench import CONFIGS, show
from tqdm import trange

from tracealbe import DiffAEModel
#import tvm # from source, v0.9.0

# Export
#model: DiffAEModel = CONFIGS['cpu_traced']('ffhq256', B=1, export=True)
#model: DiffAEModel = CONFIGS['cpu_fused']('ffhq256', B=1, export=True)
#model: DiffAEModel = CONFIGS['cuda_traced_fp16']('ffhq256', B=2, export=True)
#model: DiffAEModel = CONFIGS['cpu']('ffhq256')
lat_init = 'ckpts/ffhq256_lat_init.onnx'
lat_step = 'ckpts/ffhq256_lat.onnx'
lat_step_fused = 'ckpts/ffhq256_lat_fused.onnx'
lat_norm = 'ckpts/ffhq256_lat_norm.onnx'
img_init = 'ckpts/ffhq256_img_init.onnx'
img_step = 'ckpts/ffhq256_img.onnx'
img_step_fused = 'ckpts/ffhq256_img_fused.onnx'

# In order of preference
backends = ['CPUExecutionProvider']

# Run
B = 1
seed = 0
T = np.array([10]*B, dtype=np.int64).reshape(-1, 1)
x_lat = RandomState(seed).randn(1, 512).astype(np.float32)
x_img = RandomState(seed).randn(1, 3, 256, 256).astype(np.float32)

def run_full():
    global x_lat, x_img
    
    t0 = time.time()
    sess = ort.InferenceSession(lat_init, providers=backends)
    lat_params = sess.run(input_feed={ 'T': T },
        output_names=[
            'timestep_map',
            'alphas_cumprod',
            'alphas_cumprod_prev',
            'sqrt_recip_alphas_cumprod',
            'sqrt_recipm1_alphas_cumprod',
        ])

    lat_params[0] = lat_params[0].astype(np.int64)

    # TODO: timestep_map: keep size=1000, just pad with invalid zeros at end?
    sess = ort.InferenceSession(lat_step, providers=backends)
    for i in range(T.item()):
        x_lat = sess.run(input_feed={
            't': (T - i - 1),
            'x': x_lat,
            'timestep_map': lat_params[0],
            'alphas_cumprod': lat_params[1],
            'alphas_cumprod_prev': lat_params[2],
            'sqrt_recip_alphas_cumprod': lat_params[3],
            'sqrt_recipm1_alphas_cumprod': lat_params[4],
        }, output_names=['lats'])[0]

    sess = ort.InferenceSession(lat_norm, providers=backends)
    lats = sess.run(input_feed={'lats_in': x_lat}, output_names=['lats'])[0]

    t1 = time.time()

    sess = ort.InferenceSession(img_init, providers=backends)
    img_params = sess.run(input_feed={ 'T': T },
        output_names=[
            'timestep_map',
            'alphas_cumprod',
            'alphas_cumprod_prev',
            'sqrt_recip_alphas_cumprod',
            'sqrt_recipm1_alphas_cumprod',
        ])

    img_params[0] = img_params[0].astype(np.int64)

    sess = ort.InferenceSession(img_step, providers=backends)
    for i in trange(T.item()):
        x_img = sess.run(input_feed={
            't': (T - i - 1),
            'x': x_img,
            'lats': lats,
            'timestep_map': img_params[0],
            'alphas_cumprod': img_params[1],
            'alphas_cumprod_prev': img_params[2],
            'sqrt_recip_alphas_cumprod': img_params[3],
            'sqrt_recipm1_alphas_cumprod': img_params[4],
        }, output_names=['output'])[0]
    
    t2 = time.time()

    vlat = T.item() / (t1 - t0)
    vimg = T.item() / (t2 - t1)

    print(f'[Split] Lat: {vlat:.2f}it/s, Img: {vimg:.2f}it/s')
    return x_img

def run_fused():
    global x_lat, x_img

    t0 = time.time()
    sess = ort.InferenceSession(lat_step_fused, providers=backends)
    for i in range(T.item()):
        x_lat = sess.run(
            output_names=['lats'],
            input_feed={
                'T': T,
                't': T - i - 1,
                'x': x_lat
            })[0]

    sess = ort.InferenceSession(lat_norm, providers=backends)
    lats = sess.run(input_feed={'lats_in': x_lat}, output_names=['lats'])[0]

    t1 = time.time()
    sess = ort.InferenceSession(img_step_fused, providers=backends)
    for i in trange(T.item()):
        x_img = sess.run(
            output_names=['output'],
            input_feed={
                'T': T,
                't': T - i - 1,
                'x': x_img,
                'lats': lats,
            })[0]

    t2 = time.time()
    vlat = T.item() / (t1 - t0)
    vimg = T.item() / (t2 - t1)

    print(f'[Fused] Lat: {vlat:.2f}it/s, Img: {vimg:.2f}it/s')
    return x_img


show(run_fused())
show(run_full())
print('Done')
