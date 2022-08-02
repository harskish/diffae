# Use conda env tvm (py38)

from sympy import comp
import torch
import numpy as np
import onnxruntime as ort # pip install onnxruntime
import onnx # from src, whl/onnx. Also: conda install protobuf
from bench import CONFIGS, show
from tqdm import trange

from tracealbe import DiffAEModel
#import tvm # from source, v0.9.0

def compare(ta, tb, rtol=1e-3):
    for i, (a, b) in enumerate(zip(ta, tb)):
        if torch.is_tensor(a):
            a = a.cpu().numpy()
        if torch.is_tensor(b):
            b = b.cpu().numpy()
        
        if not np.allclose(a, b, rtol=rtol):
            print(f'i={i}: differs!')
            print(a.reshape(-1)[i:i+4])
            print(b.reshape(-1)[i:i+4])
            raise RuntimeError()

# Export
#model: DiffAEModel = CONFIGS['cpu_traced']('ffhq256', export=True)
model: DiffAEModel = CONFIGS['cpu']('ffhq256')
lat_init = 'ffhq256_lat_init.onnx'
lat_step = 'ffhq256_lat.onnx'
img_init = 'ffhq256_img_init.onnx'
img_step = 'ffhq256_img.onnx'

# In order of preference
backends = ['CPUExecutionProvider']

# IMG STEP: broken?
# mod = onnx.load(img_step)
# onnx.checker.check_model(mod)
# print(onnx.helper.printable_graph(mod.graph))

# Run
T = np.array([5], dtype=np.int64)
x = np.random.randn(1, 512).astype(np.float32)

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
ref = model.lat_sampl.forward(torch.from_numpy(T))
compare(lat_params, ref)
print('Lat-samp params match')

# TODO: timestep_map: keep size=1000, just pad with invalid zeros at end?

sess = ort.InferenceSession(lat_step, providers=backends)
for i in range(T.item()):
    x_out = sess.run(input_feed={
        't': (T - i - 1),
        'x': x,
        'timestep_map': lat_params[0],
        'alphas_cumprod': lat_params[1],
        'alphas_cumprod_prev': lat_params[2],
        'sqrt_recip_alphas_cumprod': lat_params[3],
        'sqrt_recipm1_alphas_cumprod': lat_params[4],
    }, output_names=['lats'])
    eval_model = lambda x, t: model.lat_net(x, t)
    ref = model.lat_sampl.sample_incr(torch.from_numpy(T - i - 1), torch.from_numpy(x), eval_model, *(torch.from_numpy(v) for v in lat_params))
    compare(x_out, [ref], rtol=1e-2)
    print('.', end='')
    x = x_out[0]

print('latent diffusions match')
lats = x

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
compare(img_params, model.img_sampl.forward(torch.from_numpy(T)))
print('Img-samp params match')

# Get initial from torch

# TODO: is this missing?
lats = model.lat_denorm(lats)

# TODO: needs lats from PT to work...
lats = model.sample_lat_loop(torch.from_numpy(T), torch.from_numpy(x)).cpu().numpy()

x = np.random.randn(1, 3, 256, 256).astype(np.float32)
sess = ort.InferenceSession(img_step, providers=backends)
for i in trange(T.item()):
    x_out = sess.run(input_feed={
        't': (T - i - 1),
        'x': x,
        'lats': lats,
        'timestep_map': img_params[0],
        'alphas_cumprod': img_params[1],
        'alphas_cumprod_prev': img_params[2],
        'sqrt_recip_alphas_cumprod': img_params[3],
        'sqrt_recipm1_alphas_cumprod': img_params[4],
    }, output_names=['output'])
    
    #eval_model = lambda x, t: model.img_net(x, t, cond=torch.from_numpy(lats))
    #ref = model.img_sampl.sample_incr(torch.from_numpy(T - i - 1), torch.from_numpy(x), eval_model, *(torch.from_numpy(v) for v in img_params))
    
    #compare(x_out, [ref], rtol=1)
    #print(x_out[0].reshape(-1)[i:i+4])
    #print(ref.numpy().reshape(-1)[i:i+4])
    
    x = x_out[0]

print('Results match')
show(x)
print('Done')
