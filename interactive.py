import os
import imgui
import torch
import argparse
from multiprocessing import Lock
from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache
from enum import Enum
from viewer.toolbar_viewer import ToolbarViewer
from viewer.utils import reshape_grid
from typing import Dict, Tuple
from os import makedirs
import time
from pathlib import Path
import gdown

from templates import *
from templates_latent import *
from config import *
from dataset import *
from dist_utils import *
from lmdb_writer import *
from metrics import *
from renderer import *

args = None

def sample_seeds(N, base=None):
    if base is None:
        base = np.random.randint(np.iinfo(np.int32).max - N)
    return [(base + s) for s in range(N)]

def sample_normal(shape=(1, 512), seed=None):
    seeds = sample_seeds(shape[0], base=seed)
    return torch.tensor(seeds_to_samples(seeds, shape))

def seeds_to_samples(seeds, shape=(1, 512)):
    latents = np.zeros(shape, dtype=np.float32)
    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        latents[i] = rng.standard_normal(shape[1:])
    
    return latents

class ModelViz(ToolbarViewer):    
    def __init__(self, name, batch_mode=False):
        self.batch_mode = batch_mode
        super().__init__(name, batch_mode=batch_mode)
    
    # Check for missing type annotations (break by-value comparisons)
    def check_dataclass(self, obj):
        for name in dir(obj):
            if name.startswith('__'):
                continue
            if name not in obj.__dataclass_fields__:
                raise RuntimeError(f'[ERR] Unannotated field: {name}')

    def setup_state(self):
        self.state = UIState()
        self.state_soft = UIStateSoft()
        self.rend = RendererState()
        
        self.check_dataclass(self.state)
        self.check_dataclass(self.state_soft)
        self.check_dataclass(self.rend)

        self.G_lock = Lock()
        self.rend.img_cache = {}
        self.rend.lat_cache = {}
    
    @lru_cache()
    def init_model(self):
        conf = ffhq256_autoenc_latent()
        
        conf.seed = None
        conf.pretrain = None

        model = LitModel(conf)
        state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)
        model = model.to('cuda')

        return model

    # Progress bar below images
    def draw_output_extra(self):
        self.rend.i = imgui.slider_int('', self.rend.i, 0, self.rend.last_ui_state.T)[1]

    def compute(self):
        # Copy for this frame
        s = deepcopy(self.state)

        # Perform computation
        # Detect changes
        # Only works for fields annotated with type (e.g. sliders: list)
        if self.rend.last_ui_state != s:
            self.rend.last_ui_state = s
            self.rend.model = self.init_model()
            
            # Setup samplers
            self.rend.model.conf.T_eval = max(2, s.T)
            self.rend.model.conf.latent_T_eval = max(2, s.lat_T) # not used?
            self.rend.sampl = self.rend.model.conf._make_diffusion_conf(max(2, s.T)).make_sampler()
            self.rend.lat_sampl = self.rend.model.conf._make_latent_diffusion_conf(max(2, s.lat_T)).make_sampler()

            assert self.rend.model.conf.train_mode == TrainMode.latent_diffusion
            assert self.rend.model.conf.model_type.has_autoenc()
            res = self.rend.model.conf.img_size
            self.rend.i = 0
            self.rend.intermed = sample_normal((s.B, 3, res, res), s.seed).cuda() # spaial noise

        # Check if work is done
        if self.rend.i >= s.T:
            return None

        conf = self.rend.model.conf
        ema_model = self.rend.model.ema_model

        # Sample latents
        lats = []
        for i in range(s.B):
            key = (s.seed + i, s.lat_T)
            if key not in self.rend.lat_cache:
                latent_noise = sample_normal((1, conf.style_ch), s.seed + i).cuda()
                lat = self.rend.lat_sampl.sample(
                    model=ema_model.latent_net,
                    noise=latent_noise,
                    clip_denoised=conf.latent_clip_sample,
                    progress=False,
                )

                if conf.latent_znormalize:
                    lat = lat * self.rend.model.conds_std.cuda() + self.rend.model.conds_mean.cuda()
                
                self.rend.lat_cache[key] = lat
            lats.append(self.rend.lat_cache[key])

        # Run diffusion one step forward
        model_kwargs = {'x_start': None, 'cond': torch.cat(lats, dim=0)}
        t = th.tensor([s.T - self.rend.i - 1] * s.B, device='cuda', requires_grad=False)
        ret = self.rend.sampl.ddim_sample(
            ema_model,
            self.rend.intermed,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
            eta=0.0,
        )
        self.rend.intermed = ret['sample']

        # Move on to next batch
        self.rend.i += 1

        # Read from or write to cache
        finished = [False]*s.B
        for i, img in enumerate(self.rend.intermed):
            key = (s.seed + i, s.T, s.lat_T)
            if key in self.rend.img_cache:
                self.rend.intermed[i] = torch.from_numpy(self.rend.img_cache[key]).cuda()
                finished[i] = True
            elif self.rend.i >= s.T:
                self.rend.img_cache[key] = img.cpu().numpy()

        # Early exit
        if all(finished):
            self.rend.i = s.T
        
        # Output updated grid
        return reshape_grid(0.5 * (self.rend.intermed + 1)) # => HWC
    
    def draw_toolbar(self):
        s = self.state
        s.B = imgui.input_int('B', s.B)[1]
        s.seed = max(0, imgui.input_int('Seed', s.seed, s.B, 1)[1])
        s.T = imgui.input_int('T_img', s.T, 1, 10)[1]
        s.lat_T = imgui.input_int('T_lat', s.lat_T, 1, 10)[1]

class VizMode(int, Enum):
    SINGLE = 0 # single image

# Volatile state: requires recomputation of results
@dataclass
class UIState:
    pkl: str = None
    T: int = 10
    lat_T: int = 10
    seed: int = 0
    B: int = 1

# Non-volatile (soft) state: does not require recomputation
@dataclass
class UIStateSoft:
    video_mode: VizMode = VizMode.SINGLE

@dataclass
class RendererState:
    last_ui_state: UIState = None # Detect changes in UI, restart rendering
    model: LitModel = None
    sampl: SpacedDiffusionBeatGans = None
    lat_sampl: SpacedDiffusionBeatGans = None
    intermed: torch.Tensor = None
    img_cache: Dict[Tuple[int, int, int], torch.Tensor] = None
    lat_cache: Dict[Tuple[int, int], torch.Tensor] = None
    i: int = 0 # current computation progress

def init_torch():
    # Go fast
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    
    # Stay safe
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def download_models():
    files = [
        ('1EW94TP50IKqGPbPCL-Uld1TGnRtdH8aa', 'ffhq256_autoenc_latent/last.ckpt'),
        ('1y3cGbCIuMiGDyC6S-vt0SfoAxGP-GMPN', 'ffhq256_autoenc/latent.pkl')
    ]
    for id, suffix in files:
        pth = Path(__file__).parent / 'checkpoints' / suffix
        if not pth.is_file():
            makedirs(pth.parent, exist_ok=True)
            gdown.download(id=id, output=str(pth), quiet=False)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='DiffAE visualizer')
    # parser.add_argument('input', type=str, help='Model ckpt')
    # args = parser.parse_args()

    download_models()
    init_torch()
    viewer = ModelViz('diffae_viewer')
    print('Done')
