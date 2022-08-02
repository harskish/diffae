# Benchmark DiffAE performance on Apple M2 using different backends
# - Apache TVM (w/ optimizations)
# - coremltools (using NE)
# - Onnxruntime (CoreML, w/ NE?) - only C++?

import torch
import numpy as np
from numpy.random import RandomState
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
    dev_lat = model.dev_lat
    dev_img = model.dev_img
    
    # Warmup
    _x1 = torch.randn((B, 512)).to(dev_lat)
    _x2 = torch.randn((B, 3, model.res, model.res)).to(dev_img)
    _t = torch.tensor([3] * B)
    print('Warmup', end='')
    for _ in range(3):
        _ = model.forward(_t.to(dev_lat), _t.to(dev_img), _x1, _x2)
        print('.', end='')
    print('done')

    t0 = time.time()
    latent_noise = torch.tensor(RandomState(seed).randn(B, 512), dtype=torch.float32).to(dev_lat)
    T = torch.tensor([steps_lat], dtype=torch.int64, device=dev_lat)
    lats = model.sample_lat_loop(T, latent_noise).to(dev_img)
    t1 = time.time()
    
    # Initial value: spaial noise
    intermed = torch.tensor(RandomState(seed).randn(B, 3, model.res, model.res), dtype=torch.float32).to(dev_img)
    intermed = model.sample_img_loop(T, intermed, lats)
    t2 = time.time()
    
    it_lat = (t1 - t0) / steps_lat
    it_img = (t2 - t1) / steps
    it_tot = (t2 - t0) / steps

    # Summary
    dev_desc = []
    for d, mod in zip([dev_lat, dev_img], [model.lat_net.mode, model.img_net.mode]):
        dev_desc.append(f'[{mod}]{d.type.upper()}')
    summary = f'{"+".join(dev_desc)} - Lat: {1/it_lat:.2f}it/s, img: {1/it_img:.2f}it/s, tot: {1/it_tot:.2f}it/s'
    
    if verbose:
        print(summary)

    return intermed, summary

# Fusing only makes sense for traced models
def _get_model(dev_lat, dev_img, dset, lat_fused=False, img_fused=False):
    model = DiffAEModel(dset, dev_lat, dev_img, lat_fused=lat_fused, img_fused=img_fused)
    setattr(model.lat_net, 'mode', 'INT')
    setattr(model.img_net, 'mode', 'INT')
    return model

def model_torch(dev_lat, dev_img, dset):
    return _get_model(dev_lat, dev_img, dset, False, False)

# https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/
def model_torch_traced(dev_lat, dev_img, dset, B=1, lat_fused=False, img_fused=False, export=False) -> DiffAEModel:
    model = _get_model(dev_lat, dev_img, dset, lat_fused, img_fused)
    check_trace = 'mps' not in [dev_img, dev_lat]
    
    if is_set('USE_MKLDNN') and is_set('BUILD_ONEDNN_GRAPH'):
        torch.jit.enable_onednn_fusion(True)

    T = torch.tensor([10] * B, dtype=torch.int64).to(dev_lat)
    t = T - 1

    # torch.jit.trace(lambda) => TraceFunction
    # torch.jit.trace(module.fun) => trace doesn't support compiling individual module's functions
    # torch.jit.trace_module(module, {'forward': (i1, i2)}) => Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults
    # torch.jit.trace(module) = torch.jit.trace(module.forward)?
    # torch.jit.trace_module(module) => TraceModule?
    # 'Iterating over a tensor might cause the trace to be incorrect.'

    # example_outputs no longer exists in torch 1.11+

    # Latent net init
    if not model.lat_fused:
        fwd_fun = lambda T : model.lat_sampl.forward(T)
        jit_lat_init = torch.jit.trace(fwd_fun, (T), check_trace=check_trace)
        if export:
            jit_lat_init.save(f'{dset}_lat_init.pt')
            torch.onnx.export(
                jit_lat_init,                 # model being run
                (T),                          # model input (or a tuple for multiple inputs)
                f'{dset}_lat_init.onnx',      # where to save the model (can be a file or file-like object)
                input_names = ['T'],          # the model's input names
                output_names = [              # the model's output names
                    'timestep_map',
                    'alphas_cumprod',
                    'alphas_cumprod_prev',
                    'sqrt_recip_alphas_cumprod',
                    'sqrt_recipm1_alphas_cumprod',
                ]
            )

        # Replace with jitted
        model.lat_sampl.forward = jit_lat_init

    # Latent net
    x0 = torch.randn(B, 512).to(dev_lat)
    jit_lat = None
    input_names = []
    model_name = ''
    ex_inputs = []
    
    if model.lat_fused:
        fwd_fun = lambda T, t, x : model.sample_lat_incr_fused(T, t, x)
        ex_inputs = (T, t, x0)
        input_names = ['T', 't', 'x']
        jit_lat = torch.jit.trace(fwd_fun, ex_inputs, check_trace=check_trace)
        model_name = f'{dset}_lat_fused'
        model.sample_lat_incr_fused = jit_lat
    else:
        fwd_fun = lambda t, x, *vs : model.lat_sampl.sample_incr(t, x, model.lat_net, *vs)
        ex_inputs = (t, x0, *model.lat_sampl.forward(T))
        input_names = ['t', 'x', 'timestep_map', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod']
        jit_lat = torch.jit.trace(fwd_fun, ex_inputs, check_trace=check_trace)
        model_name = f'{dset}_lat'
        #def new_fwd_lat(t, x, eval_model, *params):
        #    return jit_lat(t, x, *params)
        model.lat_sampl.sample_incr = lambda t, x, _, *params : jit_lat(t, x, *params)

    if export:
        jit_lat.save(f'{model_name}.pt')
        torch.onnx.export(
            jit_lat,                  # model being run
            ex_inputs,                # model input (or a tuple for multiple inputs)
            f'{model_name}.onnx',     # where to save the model (can be a file or file-like object)
            input_names = input_names,
            output_names = ['lats'],
            do_constant_folding=False
        )

    # Image net init
    if not model.img_fused:
        T = T.to(dev_img)
        fwd_fun = lambda T : model.img_sampl.forward(T)
        jit_img_init = torch.jit.trace(fwd_fun, (T), check_trace=check_trace)
        if export:
            jit_img_init.save(f'{dset}_img_init.pt')
            torch.onnx.export(
                jit_img_init,                 # model being run
                (T),                          # model input (or a tuple for multiple inputs)
                f'{dset}_img_init.onnx',      # where to save the model (can be a file or file-like object)
                input_names = ['T'],          # the model's input names
                output_names = [              # the model's output names
                    'timestep_map',
                    'alphas_cumprod',
                    'alphas_cumprod_prev',
                    'sqrt_recip_alphas_cumprod',
                    'sqrt_recipm1_alphas_cumprod',
                ]
            )

        # Replace with jitted
        model.img_sampl.forward = jit_img_init

    # Image net
    jit_img = None
    input_names = []
    model_name = ''
    ex_inputs = []

    lats = torch.randn(B, 512).to(dev_img)
    T = T.to(dev_img)
    t = t.to(dev_img)
    x0 = torch.randn(B, 3, model.res, model.res).to(dev_img)
    
    if model.img_fused:
        fwd_fun = lambda T, t, x, lats : model.sample_img_incr_fused(T, t, x, lats)
        ex_inputs = (T, t, x0, lats)
        input_names = ['T', 't', 'x', 'lats']
        jit_img = torch.jit.trace(fwd_fun, ex_inputs, check_trace=check_trace)
        model_name = f'{dset}_img_fused'
        model.sample_img_incr_fused = jit_img
    else:
        fwd_fun = lambda t, x, lats, *vs : model.img_sampl.sample_incr(t, x, partial(model.img_net, cond=lats), *vs)
        ex_inputs = (t, x0, lats, *model.img_sampl.forward(T))
        input_names = ['t', 'x', 'lats', 'timestep_map', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod']
        jit_img = torch.jit.trace(fwd_fun, ex_inputs, check_trace=check_trace)
        model_name = f'{dset}_img'
        #def new_fwd_img(t, x, eval_model, *params):
        #    return jit_img(t, x, eval_model.keywords['cond'], *params)
        model.img_sampl.sample_incr = lambda t, x, mod, *params : jit_img(t, x, mod.keywords['cond'], *params)

    if export:
        jit_img.save(f'{model_name}.pt')
        torch.onnx.export(
            jit_img,                     # model being run
            ex_inputs,                   # model input (or a tuple for multiple inputs)
            f'{model_name}.onnx',        # where to save the model (can be a file or file-like object)
            input_names = input_names,
            output_names = ['output'],
            do_constant_folding=False,
            keep_initializers_as_inputs=False,
            export_modules_as_functions=False,
            verbose=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX, # ONNX_ATEN, ONNX_ATEN_FALLBACK
        )

    setattr(model.lat_net, 'mode', 'JIT')
    setattr(model.img_net, 'mode', 'JIT')
    return model

def load_pt_model(dev_lat, dev_img, dset):
    model = DiffAEModel(dset, dev_lat, dev_img)

    model.lat_sampl.forward = torch.jit.load(f'{dset}_lat_init.pt').to(dev_lat)
    model.img_sampl.forward = torch.jit.load(f'{dset}_img_init.pt').to(dev_img)
    model.lat_sampl.sample_incr = torch.jit.load(f'{dset}_lat.pt').to(dev_lat)
    model.img_sampl.sample_incr = torch.jit.load(f'{dset}_img.pt').to(dev_img)

    setattr(model.lat_net, 'mode', 'PTH')
    setattr(model.img_net, 'mode', 'PTH')
    return model

from functools import partial
CONFIGS = {
    'cuda': partial(model_torch, 'cuda', 'cuda'),
    'cuda_traced': partial(model_torch_traced, 'cuda', 'cuda'),
    'cuda_pt': partial(load_pt_model, 'cuda', 'cuda'),
    'cpu': partial(model_torch, 'cpu', 'cpu'),
    'cpu_traced': partial(model_torch_traced, 'cpu', 'cpu'),
    'cpu_pt': partial(load_pt_model, 'cpu', 'cpu'),
    'mps': partial(model_torch, 'mps', 'mps'),
    'mps_fused': partial(model_torch_traced, 'mps', 'mps', lat_fused=True, img_fused=True, export=False),
    'mps_traced': partial(model_torch_traced, 'mps', 'mps'),
    'mps_pt': partial(load_pt_model, 'mps', 'mps'),
    'm2_opt': partial(model_torch_traced, 'cpu', 'mps'),
}

if __name__ == '__main__':
    dset = 'ffhq256'
    mps = getattr(torch.backends, 'mps', None)
    seed = random.randint(0, 1<<32-1)
    
    configs = []
    if torch.cuda.is_available():
        configs.append('cuda')
        configs.append('cuda_traced')
        configs.append('cuda_pt')
    
    if mps and mps.is_available() and mps.is_built():
        configs.append('mps_traced')
        configs.append('mps')
        configs.append('mps_fused')
        configs.append('mps_pt')
        configs.append('m2_opt')
    
    configs.append('cpu')
    configs.append('cpu_traced')
    configs.append('cpu_pt')

    summaries = []
    for cfg in configs:
        print('Running', cfg)
        img, summ = run(CONFIGS[cfg](dset), seed=seed)
        summaries.append(summ)
        show(img)

    print('Results:')
    for conf, summ in zip(configs, summaries):
        print(f'{conf}: {summ}')

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

    # Torch-only rewrite (1.8.2022)

    # M1 Pro MacBook Pro 14" (2E+6P+14GPU) - no debugger
    # cpu         Lat:  90.62it/s, img: 0.50it/s, tot: 0.50it/s
    # cpu_traced  Lat: 118.95it/s, img: 0.52it/s, tot: 0.52it/s
    # mps         Lat:  71.70it/s, img: 4.41it/s, tot: 4.15it/s
    # mps_traced  Lat: 107.92it/s, img: 4.44it/s, tot: 4.26it/s
    # m2_opt      Lat:  88.16it/s, img: 4.35it/s, tot: 4.14it/s
    
    # M1 Pro MacBook Pro 14" (2E+6P+14GPU) - with debugger
    # cpu         Lat:  51.82it/s, img: 0.50it/s, tot: 0.49it/s
    # cpu_traced  Lat: 121.03it/s, img: 0.50it/s, tot: 0.50it/s
    # mps         Lat:  51.25it/s, img: 4.28it/s, tot: 3.95it/s
    # mps_traced  Lat: 101.61it/s, img: 4.35it/s, tot: 4.17it/s
    # m2_opt      Lat:  88.10it/s, img: 4.41it/s, tot: 4.20it/s