# Benchmark DiffAE performance on Apple M2 using different backends
# - Apache TVM (w/ optimizations)
# - coremltools (using NE)
# - Onnxruntime (CoreML, w/ NE?) - only C++?

import torch
import numpy as np
from numpy.random import RandomState
from model.unet_autoenc import AutoencReturn
import templates_latent
from PIL import Image
from tqdm import trange
import random
import time
import re

from tracealbe import DiffAEModel
torch.autograd.set_grad_enabled(False)

# Repr monkey patch
_orig = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda t: _orig(t.cpu())

def show(t):
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    if np.issubdtype(t.dtype, np.floating):
        if t.min() < 0.1:
            t = 0.5*(t + 1)
        t = np.uint8(255*np.clip(t, 0, 1))
    t = t.squeeze()
    if t.ndim == 2:
        t = np.expand_dims(t, -1)
    if t.shape[0] in [1, 3, 4]: # CHW => HWC
        t = np.transpose(t, (1, 2, 0))
    Image.fromarray(t).show()

def get_build_opt(name: str):
    pattern = r'(-D)?{}=?([^\s,]*)?'.format(name)
    matches = re.findall(pattern, torch.__config__.show(), flags=re.IGNORECASE)
    vals = [m[1] for m in matches if m != ('')] # only -DFOO or FOO=BAR
    
    if not vals:
        return None # not found
    elif vals[0] == '':
        return True # defined
    else:
        return vals[0] # set to value

def is_set(name: str):
    val = str(get_build_opt(name)).upper()
    return not val in ['NONE', 'FALSE', 'OFF', '0']

def get_samplers(conf, t_img=10, t_lat=10):
    conf_img = conf._make_diffusion_conf(conf.T)
    conf_img.use_timesteps = np.linspace(0, conf.T, max(2, t_img) + 1, dtype=np.int32).tolist()[:-1]
    sampl = conf_img.make_sampler()

    conf_lat = conf._make_latent_diffusion_conf(conf.T)
    conf_lat.use_timesteps = np.linspace(0, conf.T, max(2, t_lat) + 1, dtype=np.int32).tolist()[:-1]
    lat_sampl = conf_lat.make_sampler()

    return (sampl, lat_sampl)

def run(model: DiffAEModel, steps=10, B=1, verbose=True, seed=None):
    random.seed(time.time())
    seed = random.randint(0, 1<<32-1) if seed is None else seed
    steps_lat = 10
    ran_fun = trange if verbose else range
    dev_lat = model.dev_lat
    dev_img = model.dev_img
    
    # Warmup
    _x1 = torch.randn((B, 512)).to(dev_lat)
    _x2 = torch.randn((B, 3, model.res, model.res)).to(dev_img)
    _t = torch.tensor([9] * B)
    for _ in range(3):
        _ = model.forward(_t.to(dev_lat), _t.to(dev_img), _x1, _x2)

    t0 = time.time()
    latent_noise = torch.tensor(RandomState(seed).randn(B, 512), dtype=torch.float32).to(dev_lat)
    T = torch.tensor([steps_lat], dtype=torch.int32)
    lats = model.sample_lat(T, latent_noise)
    t1 = time.time()
    
    # Initial value: spaial noise
    intermed = torch.tensor(RandomState(seed).randn(B, 3, model.res, model.res), dtype=torch.float32).to(dev_img)
    params = model.get_img_sampl_params(torch.tensor([steps], dtype=torch.int32))

    for i in ran_fun(steps):
        t = torch.tensor([steps - i - 1] * B, device=dev_img)
        intermed = model.sample_img_incr(t, intermed, lats, **params)

    t2 = time.time()
    
    it_lat = (t1 - t0) / steps_lat
    it_img = (t2 - t1) / steps
    it_tot = (t2 - t0) / steps

    if verbose:
        desc = ' [TR]' if model.traced else ''
        print(f'{dev_img.type}{desc} - Lat: {1/it_lat:.2f}it/s, img: {1/it_img:.2f}it/s, tot: {1/it_tot:.2f}it/s')

    return intermed

def model_torch(dev, dset):
    model = DiffAEModel(dset)
    setattr(model, 'traced', False)
    return model.to(dev)

# https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/
def model_torch_traced(dev, dset, B=1):
    model = model_torch(dev, dset)
    
    if is_set('USE_MKLDNN') and is_set('BUILD_ONEDNN_GRAPH'):
        torch.jit.enable_onednn_fusion(True)

    x = torch.randn((B, 512)).to(dev)
    t = torch.tensor([9] * B, device=dev) # unscaled

    # torch.jit.trace(lambda) => TraceFunction
    # torch.jit.trace(module.fun) => trace doesn't support compiling individual module's functions
    # torch.jit.trace_module(module, {'forward': (i1, i2)}) => Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults
    # torch.jit.trace(module) = torch.jit.trace(module.forward)?
    # torch.jit.trace_module(module) => TraceModule?

    # TODO: 'Iterating over a tensor might cause the trace to be incorrect.'

    def test_fun(t, x):
        #return model.get_img_sampl_params(t)['posterior_variance']
        #params = model.lat_sampl.forward(t)
        #return model.lat_sampl.sample_incr(t-1, x, model.lat_net, **params)
        return model.lat_sampl.sample(t, x, model.lat_net)
        #params = model.lat_sampl.forward(t)
        # interm = x
        # n_iter = t.item() #torch.ones(1).repeat(t).size(0)
        # for t in torch.arange(0, n_iter, device=x.device).flip(0).view(-1, 1):
        #     #interm = model.lat_sampl.sample_incr(t, interm, model.lat_net, **params)
        #     interm = interm + torch.randn_like(interm)*t
        # return interm

    #jitted = torch.jit.script(test_fun)
    jitted = torch.jit.trace(test_fun, (t, x))
    print(jitted(torch.tensor([5] * B, device=dev), x))
    print(jitted(torch.tensor([10] * B, device=dev), x))

    # Latent net
    x0_lat = torch.randn(B, 512).to(dev)
    fwd_fun = lambda t, x : model.sample_lat(t, x)
    jit_lat = torch.jit.trace(fwd_fun, (t, x0_lat), check_trace=False)
    #jit_lat = torch.jit.trace_module(model, {'sample_lat': (t, x0_lat)}, check_trace=False)
    jit_lat.save(f'{dset}_lat.pt')
    assert torch.allclose(out_ref.cpu(), jit_lat(t, x).cpu()), 'Lat-net outputs differ'
    
    # Export ONNX
    torch.onnx.export(jit_lat,                      # model being run
                      (t, x),                       # model input (or a tuple for multiple inputs)
                      f'{dset}_lat.onnx',           # where to save the model (can be a file or file-like object)
                      input_names = ['t', 'z'],     # the model's input names
                      output_names = ['output'],    # the model's output names
                      example_outputs = out_ref)
    
    # Replace with jitted
    def new_fwd_lat(t, x, **kwargs):
        return jit_lat(t, x)
    model.lat_net.forward = new_fwd_lat

    # Image diffusion
    cond = torch.randn((B, 512), device=dev)
    x = torch.randn((B, 3, model.conf.img_size, model.conf.img_size)).to(dev)
    fwd_fun = lambda x, t, cond: model.ema_model(x, t, cond=cond, x_start=None).pred
    out_ref = fwd_fun(x, t, cond)
    jit_img = torch.jit.trace(fwd_fun, (x, t, cond), check_trace=False)
    jit_img.save(f'{dset}_img.pt')
    #assert torch.allclose(out_ref.cpu(), jit_img(x, t, cond).cpu()), 'Img-net outputs differ'

    torch.onnx.export(jit_img,                          # model being run
                      (x, t, cond),                     # model input (or a tuple for multiple inputs)
                      f'{dset}_img.onnx',               # where to save the model (can be a file or file-like object)
                      input_names = ['z', 't', 'cond'], # the model's input names
                      output_names = ['output'],        # the model's output names
                      example_outputs = out_ref,
                      opset_version=12)                 # einsum

    # Replace with jitted
    def new_fwd_img(x, t, cond=None, **kwargs):
        assert not any(kwargs.values()), 'Unsupported kwargs provided'
        ret = jit_img(x, t, cond)
        return AutoencReturn(pred=ret, cond=cond)
    model.ema_model.forward = new_fwd_img

    setattr(model, 'traced', True)
    return model

# M2 optimal: mix of traced CPU and GPU
def get_model_m2_optimal(dset):
    # Image diffusion fastest on traced MPS
    model = model_torch_traced('mps', dset)
    
    # Traced latent net fastest on CPU
    model.ema_model.latent_net = model_torch_traced('cpu', dset).ema_model.latent_net

    return model

from functools import partial
CONFIGS = {
    'cuda': partial(model_torch, 'cuda'),
    'cuda_traced': partial(model_torch_traced, 'cuda'),
    'cpu': partial(model_torch, 'cpu'),
    'cpu_traced': partial(model_torch_traced, 'cpu'),
    'mps': partial(model_torch, 'mps'),
    'mps_traced': partial(model_torch_traced, 'mps'),
    'm2_opt': get_model_m2_optimal,
}

if __name__ == '__main__':
    dset = 'ffhq256'
    
    #show(run(CONFIGS['cuda'](dset)))
    #show(run(CONFIGS['cuda_traced'](dset)))
    #show(run(CONFIGS['cpu'](dset)))
    show(run(CONFIGS['cpu_traced'](dset)))
    show(run(CONFIGS['mps'](dset)))
    show(run(CONFIGS['mps_traced'](dset)))
    show(run(CONFIGS['m2_opt'](dset)))
    print('Done')

    # M2 MacBook Air 13" (4E+4P+10GPU)
    # cpu         Lat:  46.85it/s, img: 0.46it/s, tot: 0.46it/s
    # cpu_traced  Lat: 170.92it/s, img: 0.47it/s, tot: 0.47it/s
    # mps         Lat:  20.82it/s, img: 2.99it/s, tot: 2.61it/s
    # mps_traced  Lat:  34.98it/s, img: 3.01it/s, tot: 2.77it/s
    # m2_opt      Lat:  62.44it/s, img: 2.98it/s, tot: 2.84it/s

    # M1 Pro MacBook Pro 14" (2E+6P+14GPU)
    # cpu         Lat:  86.85it/s, img: 0.54it/s, tot: 0.54it/s
    # cpu_traced  Lat: 120.78it/s, img: 0.53it/s, tot: 0.53it/s
    # mps         Lat:  29.44it/s, img: 4.05it/s, tot: 3.56it/s
    # mps_traced  Lat:  40.38it/s, img: 4.19it/s, tot: 3.80it/s
    # m2_opt      Lat:  92.37it/s, img: 4.26it/s, tot: 4.07it/s (1.43x)
    # full_trace                                  tot: 4.24it/s