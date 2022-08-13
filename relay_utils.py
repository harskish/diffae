import os
import tvm
from tvm import relay
from typing import List, Tuple
from pathlib import Path

rootdir = Path(__file__).parent / 'ckpts'

def export_pt_as_relay(model_jitted, shape_list: List[Tuple[str, List[int]]], name):
    mod, params = load_pt_model(model_jitted, shape_list)

    try:
        with open((rootdir / name).with_suffix('_relay.json'), "w") as fo:
            fo.write(tvm.ir.save_json(mod))
    except RuntimeError as e:
        print(e)

    try:
        with open((rootdir / name).with_suffix('_relay.params'), "wb") as fo:
            fo.write(relay.save_param_dict(params))
    except RuntimeError as e:
        print(e)

def load_pt_model(model_jitted, shape_list: List[Tuple[str, List[int]]]):
    #shape_list = [('input_ids', [1, 128])]
    mod, params = relay.frontend.from_pytorch(model_jitted, shape_list)
    return (mod, params)
    

# def load_relay_model(name,
#                   path=".",
#                   relay_file="_pt_model.json",
#                   relay_params="_pt_model.params"):
#     with open(os.path.join(path, name + relay_file), "r") as fi:
#         mod = tvm.ir.load_json(fi.read())
#     with open(os.path.join(path, name + relay_params), "rb") as fi:
#         params = relay.load_param_dict(fi.read())
#     mod = tvm.relay.transform.FastMath()(mod)
#     mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
#     BindPass = tvm.relay.transform.function_pass(lambda fn, new_mod, ctx:
#                             tvm.relay.build_module.bind_params_by_name(fn, params), opt_level=1)
#     mod = BindPass(mod)
#     mod = tvm.relay.transform.FoldConstant()(mod)
#     mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
#     mod = tvm.relay.transform.FoldConstant()(mod)
#     return mod, params, {"input_ids" : [1, 128]}


