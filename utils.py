import sys

import torch

from functools import wraps
from time import time

from transformers import AutoModelForCausalLM


DMAP_CHOICES = ['auto', 'sequential']


DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def timed(f):
    @wraps(f)
    def timed_f(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'{f.__name__}: {end-start:.1f} seconds', file=sys.stderr)
        return result
    return timed_f


@timed
def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=DTYPE_MAP[args.dtype],
        trust_remote_code=args.trust_remote_code,
    )
    return model
