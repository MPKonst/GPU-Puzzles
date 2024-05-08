from PIL import Image
import chalk
import numpy as np
import os
import numba
from numba import cuda # for some reason, this import is necessary, otherwise numba.cuda.jit is not found
import math
from functools import partial

import io
from lib import CudaProblem, Coord
from flash_attention import (
    flash_attn_forward_kernel_factory, flash_attn_forward_no_stabilisation_kernel_factory
)

kernel_to_visualise = flash_attn_forward_kernel_factory


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def attn_spec(q, k, v):
    return softmax(q@k.T)@v


SEQ_LEN = 9
HIDDEN_DIM = 10
TPB_x = 5

SEQ_LEN = 5
HIDDEN_DIM = 6
TPB_x = 2
# np.random.seed(78)
q = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32) * np.random.rand()
k = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32) * np.random.rand()
v = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32) * np.random.rand()

final_out = np.zeros((SEQ_LEN, HIDDEN_DIM))

problem = CudaProblem(
    "Flash Attention",
    partial(kernel_to_visualise, tpb_x=TPB_x, hidden_dim=HIDDEN_DIM),
    inputs=[q, k, v],
    out=final_out,
    blockspergrid=Coord((SEQ_LEN + TPB_x - 1) // TPB_x, 1),
    threadsperblock=Coord(TPB_x, HIDDEN_DIM),
    spec=attn_spec,
    input_names=["q", "k", "v"],
)
problem.show(
    sparse=False, svg_height_factor=5
)