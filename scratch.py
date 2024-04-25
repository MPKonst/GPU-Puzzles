from PIL import Image
import chalk
import numpy as np
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
import numba
from numba import cuda


import io
from lib import CudaProblem, Coord


def render_diagram(diagram: chalk.Diagram):
    buff = io.BytesIO()
    diagram.render(buff)
    buff.seek(0)
    pil_image = Image.open(buff)
    del buff
    return pil_image

## Attention

# def attn_spec(q, k, v):
#     logits = q @ k.T
#     # probs = np.exp(logits)
#     # probs /= probs.sum(axis=1, keepdims=True)
#     return logits # probs @ v

# TPB = 3
# def attn_test(cuda):
#     def attn_kernel(out, q, k, v, seq_len, hidden_dim) -> None:
#         local_i = cuda.threadIdx.x
#         local_j = cuda.threadIdx.y
#         i = cuda.blockIdx.x * cuda.blockDim.x + local_i
#         j = cuda.blockIdx.y * cuda.blockDim.y + local_j
#         # compute attention logits as q @ k.T
#         q_shared = cuda.shared.array((TPB, TPB), numba.float32)
#         k_T_shared = cuda.shared.array((TPB, TPB), numba.float32)
#         total = 0
#         num_block_columns = (hidden_dim + cuda.blockDim.y - 1) // cuda.blockDim.y
#         for shift in range(num_block_columns):
#             shifted_i = local_i + shift * cuda.blockDim.x # 2, 5, 8
#             shifted_j = local_j + shift * cuda.blockDim.y # 0, 3, 6
#             if i < seq_len and shifted_j < hidden_dim:
#                 q_shared[local_i, local_j] = q[i, shifted_j]
#             if shifted_i < hidden_dim and j < seq_len:
#                 k_T_shared[local_i, local_j] = k[j, shifted_i]
#             cuda.syncthreads()
#             # compute the inner productinside each block
#             a_subblock_width = cuda.blockDim.y
#             # if we are reading the last block column of a,
#             # it could have fewer columns that the loop should run through
#             if shift == num_block_columns - 1:
#                 a_subblock_width = hidden_dim - (shift * cuda.blockDim.y)
#             for local_idx in range(a_subblock_width):
#                 total += q_shared[local_i, local_idx] * k_T_shared[local_idx, local_j]
#             cuda.syncthreads()
#         if i < seq_len and j < seq_len:
#             out[i, j] = total
#     return attn_kernel


# SEQ_LEN = 10
# HIDDEN_DIM = 5
# out = np.zeros((SEQ_LEN, SEQ_LEN))
# q = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(
#     (SEQ_LEN, HIDDEN_DIM)
# )
# k = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(
#     (SEQ_LEN, HIDDEN_DIM)
# )
# v = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(
#     (SEQ_LEN, HIDDEN_DIM)
# )

# # since our output is seq_len x seq_len, that's the dimension we need to tile
# # with thread blocks.
# THREADSPERBLOCK = Coord(TPB, TPB)
# BLOCKSPERGRID = Coord((SEQ_LEN + TPB - 1) // TPB, (SEQ_LEN + TPB - 1) // TPB)
# problem = CudaProblem(
#     "Attention",
#     attn_test,
#     [q, k, v],
#     out,
#     args = [SEQ_LEN, HIDDEN_DIM],
#     blockspergrid=BLOCKSPERGRID,
#     threadsperblock=THREADSPERBLOCK,
#     spec=attn_spec,
# )
# # problem.show(sparse=True)


def rowwise_sum_spec(m):
    return m.sum(axis=1) 

TPB = 3
def rowwise_sum_test(cuda):
    def rowwise_sum_kernel(out, m):
        rows, cols = m.shape
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        i = cuda.blockIdx.x * cuda.blockDim.x + local_i
        j = cuda.blockIdx.y * cuda.blockDim.y + local_j

        shared = cuda.shared.array((TPB, TPB), numba.float32)
        # copy into shared memory (performing one summation)
        if 2 * j < cols and i < rows:
            shared[local_i, local_j] = m[i, 2 * j]
        cuda.syncthreads()
        if 2 * j + 1 < cols and i < rows:
            shared[local_i, local_j] = shared[local_i, local_j] + m[i, 2 * j + 1]
        cuda.syncthreads()

        # perform the parallel scan
        q = cuda.blockDim.x // 2
        k = 1
        while q:
            if not local_j % (2 * k):
                position_to_add = 2 * (j + k)
                if position_to_add < cols and i < rows:
                    shared[local_i, local_j] = shared[local_i, local_j] + shared[local_i, local_j + k]
            k *= 2
            q = q // 2
            cuda.syncthreads()
        # write the result to global memory
        if local_j == 0 and 2 * j < cols and i < rows:
            out[i, cuda.blockIdx.y] = shared[local_i, 0]
        cuda.syncthreads()
    return rowwise_sum_kernel


ROW_SIZE = 10
COL_SIZE = 5
m = np.arange(ROW_SIZE * COL_SIZE).reshape(
    (ROW_SIZE, COL_SIZE)
)

THREADSPERBLOCK = Coord(TPB, TPB)
BPG_x = (ROW_SIZE + TPB - 1) // TPB
BPG_y = (COL_SIZE + 2 * TPB - 1) // (2 * TPB)
BLOCKSPERGRID = Coord(BPG_x, BPG_y)

out = np.zeros((ROW_SIZE, BLOCKSPERGRID.y)) # we'll need to write blocks to global memory and read them back in
problem = CudaProblem(
    "Row-wise sum",
    rowwise_sum_test,
    [m],
    out,
    blockspergrid=BLOCKSPERGRID,
    threadsperblock=THREADSPERBLOCK,
    spec=rowwise_sum_spec,
)
problem.show(sparse=True)


# def multi_block_sum_1d_test(cuda, tpb):
#     """
#     The following attempts to implement a parallel scan summation for a 1-d array.
#     However, it's unreliable because CUDA does not have bewteen-block synchronization.
#     """
#     def multi_block_sum_1d(out, m, size):
#         local_i = cuda.threadIdx.x
#         i = cuda.blockIdx.x * cuda.blockDim.x + local_i
#         shared = cuda.shared.array(tpb, numba.float32)
#         # within-block parallel scan
#         if 2 * i < size:
#             shared[local_i] = m[2 * i]
#         if 2 * i + 1 < size:
#             shared[local_i] = shared[local_i] + m[2 * i + 1]
#         q = cuda.blockDim.x // 2
#         k = 1
#         while q:
#             if not local_i % (2 * k):
#                 position_to_add = 2 * (i + k)
#                 if position_to_add < size:
#                     shared[local_i] = shared[local_i] + shared[local_i + k]
#             k *= 2
#             q = q // 2
#             cuda.syncthreads()
#         if local_i == 0 and 2 * i < size:
#             out[cuda.blockIdx.x] = shared[0]
#         cuda.syncthreads()
#         # we now re-read each within-block sum to  
#         blocks_that_wrote = (size + 2 * cuda.blockDim.x - 1) // (2 * cuda.blockDim.x)
#         while blocks_that_wrote > 1: # blocks_that_wrote becomes the new "size"
#             if 2 * i < blocks_that_wrote:
#                 shared[local_i] = out[2 * i]
#             if 2 * i + 1 < blocks_that_wrote:
#                 shared[local_i] = shared[local_i] + out[2 * i + 1]
#             q = cuda.blockDim.x // 2
#             k = 1
#             while q:
#                 if not local_i % (2 * k):
#                     position_to_add = 2 * (i + k)
#                     if position_to_add < blocks_that_wrote:
#                         shared[local_i] = shared[local_i] + shared[local_i + k]
#                 k *= 2
#                 q = q // 2
#                 cuda.syncthreads()
#             if local_i == 0 and 2 * i < blocks_that_wrote:
#                 out[cuda.blockIdx.x] = shared[0]
#             cuda.syncthreads()
#             blocks_that_wrote = (
#                 (blocks_that_wrote + (2 * cuda.blockDim.x) - 1) // (2 * cuda.blockDim.x)
#             )
#         cuda.syncthreads()
#     return multi_block_sum_1d


# size = 10
# m = np.arange(size)
# tpb = 4
# bpg = (size + 2 * tpb - 1) // (2 * tpb)
# out = np.zeros(bpg)
# print(tpb, bpg)
# # problem = CudaProblem(
# #     "Multiblock sum 1-d",
# #     multi_block_sum_1d_test,
# #     [m],
# #     out,
# #     args=[size],
# #     blockspergrid=Coord(bpg, 1),
# #     threadsperblock=Coord(tpb, 1),
# #     spec=multi_block_sum_1d_spec,
# # )
# # problem.show()


# multi_block_sum_1d = cuda.jit(multi_block_sum_1d_test(cuda, tpb))
# multi_block_sum_1d[bpg, tpb](m, size, out)
   