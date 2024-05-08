from PIL import Image
import chalk
import numpy as np
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
import numba
from numba import cuda # for some reason, this import is necessary, otherwise numba.cuda.jit is not found
import math
from functools import partial

import io
from lib import CudaProblem, Coord


def render_diagram(diagram: chalk.Diagram):
    buff = io.BytesIO()
    diagram.render(buff)
    buff.seek(0)
    pil_image = Image.open(buff)
    del buff
    return pil_image

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

# Attention

def attn_logits_spec(q, k):
    logits = q @ k.T
    # probs = np.exp(logits)
    # probs /= probs.sum(axis=1, keepdims=True)
    return logits # probs @ v



def matmul_kernel_factory(cuda, tpb, transpose_a=False, transpose_b=False):
    def matmul_kernel(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        i = cuda.blockIdx.x * cuda.blockDim.x + local_i
        j = cuda.blockIdx.y * cuda.blockDim.y + local_j

        left_outer_dim, right_outer_dim = out.shape
        inner_dim = a.shape[0] if transpose_a else a.shape[1]

        a_shared = cuda.shared.array((tpb, tpb), numba.float32)
        b_shared = cuda.shared.array((tpb, tpb), numba.float32)
        total = 0
        num_block_columns = (inner_dim + cuda.blockDim.y - 1) // cuda.blockDim.y
        for shift in range(num_block_columns):
            shifted_i = local_i + shift * cuda.blockDim.x
            shifted_j = local_j + shift * cuda.blockDim.y

            if transpose_a:
                if i < a.shape[1] and shifted_j < a.shape[0]:
                    a_shared[local_i, local_j] = a[shifted_j, i]
            else:
                if i < a.shape[0] and shifted_j < a.shape[1]:
                    a_shared[local_i, local_j] = a[i, shifted_j]
            if transpose_b:
                if shifted_i < b.shape[1] and j < b.shape[0]:
                    b_shared[local_i, local_j] = b[j, shifted_i]
            else:
                if shifted_i < b.shape[0] and j < b.shape[1]:
                    b_shared[local_i, local_j] = b[shifted_i, j]
            cuda.syncthreads()
            # compute the inner product inside each block
            a_subblock_width = cuda.blockDim.y
            # if we are reading the last block column of a,
            # it could have fewer columns that the loop should run through
            if shift == num_block_columns - 1:
                a_subblock_width = inner_dim - (shift * cuda.blockDim.y)
            for local_idx in range(a_subblock_width):
                total += a_shared[local_i, local_idx] * b_shared[local_idx, local_j]
        if i < left_outer_dim and j < right_outer_dim:
            out[i, j] = total
    return matmul_kernel


def exponentiate_kernel_factory(cuda):
    def expenentiate_kernel(out, m):
        i, j = cuda.grid(2)
        if i < m.shape[0] and j < m.shape[1]:
            out[i, j] = math.exp(m[i, j])
    return expenentiate_kernel

def rowwise_div_kernel_factory(cuda):
    def rowwise_div_kernel(out, m, denoms):
        i, j = cuda.grid(2)
        if i < m.shape[0] and j < m.shape[1]:
            out[i, j] = m[i, j] / denoms[i]
    return rowwise_div_kernel

# row-wise sum with a parallel scan

def rowwise_sum_spec(m):
    return m.sum(axis=1) 

def rowwise_sum_factory(cuda, tpb):
    def rowwise_sum_kernel(out, m, rows, cols):
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        i = cuda.blockIdx.x * cuda.blockDim.x + local_i
        j = cuda.blockIdx.y * cuda.blockDim.y + local_j

        shared = cuda.shared.array((tpb, tpb), numba.float32)
        # copy into shared memory
        if i < rows and j < cols:
            shared[local_i, local_j] = m[i, j]
        cuda.syncthreads()

        # perform the parallel scan
        q = cuda.blockDim.y
        k = 1
        while q:
            if not local_j % (2 * k):
                if i < rows and j + k < cols and local_j + k < cuda.blockDim.y:
                    shared[local_i, local_j] = shared[local_i, local_j] + shared[local_i, local_j + k]
            k *= 2
            q = q // 2
            cuda.syncthreads()
        # write the result to global memory
        if local_j == 0 and j < cols and i < rows:
            out[i, cuda.blockIdx.y] = shared[local_i, 0]
        cuda.syncthreads()
    return rowwise_sum_kernel


def test_rowwise_sum():
    TPB = 3
    m = np.arange(100).reshape(10, 10)
    BPG_x = (m.shape[0] + TPB - 1) // TPB
    BPG_y = (m.shape[1] + TPB - 1) // TPB
    out = np.zeros(shape=(10, BPG_y))
    rws_kernel = numba.cuda.jit(rowwise_sum_factory(cuda, TPB))
    rws_kernel[(BPG_x, BPG_y), (TPB, TPB)](out, m, *m.shape)
    expected = np.concatenate([m[:, i: i+TPB].sum(axis=1, keepdims=True) for i in range(0, 10, TPB)], axis=1)
    np.testing.assert_allclose(out, expected)

def test_matmul():
    TPB = 3
    left_dim, inner_dim, right_dim = 10, 4, 7
    for transpose_a in [False, True]:
        for transpose_b in [False, True]:
            out = np.zeros((left_dim, right_dim))
            a = np.arange(left_dim * inner_dim).reshape(
                (left_dim, inner_dim) if not transpose_a else (inner_dim, left_dim)
            )
            b = np.arange(inner_dim * right_dim).reshape(
                (inner_dim, right_dim) if not transpose_b else (right_dim, inner_dim)
            )
            expected = (a.T if transpose_a else a) @ (b.T if transpose_b else b)

            bpg_x = (left_dim + TPB - 1) // TPB
            bpg_y = (right_dim + TPB - 1) // TPB

            mtmk = numba.cuda.jit(matmul_kernel_factory(cuda, TPB, transpose_a, transpose_b))
            mtmk[(bpg_x, bpg_y), (TPB, TPB)](out, a, b)
            np.testing.assert_allclose(out, expected)


def test_attn():
    """Computing attention will need to be done with several different kernel calls

    Step 1: compute the attention logits
    Step 2: compute the row-wise maximums with several reduction kernels
    Step 3: compute the row-wise exponentials in a single kernel
    Step 4: compute the row-wise sums in several kernels
    Step 5: divide the row-wise exponentials by the row-wise sums in a single kernel
    Step 6: compute the final attention values by multiplying the row-wise softmaxes by the values
    """
    TPB = 32
    SEQ_LEN = 500
    HIDDEN_DIM = 64
    divider = 1000
    q = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(
        (SEQ_LEN, HIDDEN_DIM)
    ) / divider
    k = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(
        (SEQ_LEN, HIDDEN_DIM)
    ) / divider
    v = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(
        (SEQ_LEN, HIDDEN_DIM)
    ) / divider

    q = np.random.rand(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider
    k = np.random.rand(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider
    v = np.random.rand(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider


    THREADSPERBLOCK = (TPB, TPB)
    BPG = (SEQ_LEN + TPB - 1) // TPB
    BLOCKSPERGRID = BPG, BPG

    logits_out = np.zeros((SEQ_LEN, SEQ_LEN))
    sumexp_out = np.zeros((SEQ_LEN, BPG))
    attn_weights_out = np.zeros((SEQ_LEN, SEQ_LEN))
    final_out = np.zeros((SEQ_LEN, HIDDEN_DIM))

    logits_kernel = matmul_kernel_factory(numba.cuda, TPB, transpose_b=True)
    exp_kernel = exponentiate_kernel_factory(numba.cuda)
    rowwise_sum_kernel = rowwise_sum_factory(numba.cuda, TPB)
    rowwise_div_kernel = rowwise_div_kernel_factory(numba.cuda)
    values_kernel = matmul_kernel_factory(numba.cuda, TPB)

    numba.cuda.jit(logits_kernel)[BLOCKSPERGRID, THREADSPERBLOCK](
        logits_out, q, k
    )
    numba.cuda.jit(exp_kernel)[BLOCKSPERGRID, THREADSPERBLOCK](attn_weights_out, logits_out)
    np.testing.assert_allclose(np.exp(q@k.T), attn_weights_out, rtol=1e-4)

    jitted_rowwise_sum_kernel = numba.cuda.jit(rowwise_sum_kernel)

    blocks_that_wrote = SEQ_LEN
    blocks_that_will_write = (blocks_that_wrote + TPB - 1) // TPB
    sum_input = attn_weights_out
    # expected = exped
    while blocks_that_wrote > 1:
        sumexp_out = np.zeros_like(sumexp_out)
        jitted_rowwise_sum_kernel[(BPG, blocks_that_will_write), (TPB, TPB)](
            sumexp_out, sum_input, SEQ_LEN, blocks_that_wrote
        )
        # print(sumexp_out)
        # expected = np.concatenate([expected[:, i: i+TPB].sum(axis=1, keepdims=True) for i in range(0, blocks_that_wrote, TPB)], axis=1)
        # print(expected)
        # if blocks_that_wrote == 4:
        #     break
        sum_input = sumexp_out
        blocks_that_wrote = blocks_that_will_write
        blocks_that_will_write = (blocks_that_wrote + TPB - 1) // TPB
    # print(sumexp_out[:, 0])
    # print(exped.sum(axis=1))
    np.testing.assert_allclose(np.exp(q@k.T).sum(axis=1), sumexp_out[:, 0], rtol=1e-4)

    div_input = attn_weights_out
    attn_weights_out = np.zeros_like(attn_weights_out)
    numba.cuda.jit(rowwise_div_kernel)[BLOCKSPERGRID, THREADSPERBLOCK](
        attn_weights_out, div_input, sumexp_out[:, 0].copy()
    )
    # print(attn_weights_out)
    # print(scipy.special.softmax(q@k.T))
    np.testing.assert_allclose(softmax(q@k.T, axis=-1), attn_weights_out, rtol=1e-4)

    numba.cuda.jit(values_kernel)[(BPG, (HIDDEN_DIM + TPB - 1) // TPB), THREADSPERBLOCK](
        final_out, attn_weights_out, v
    )
    np.testing.assert_allclose(softmax(q@k.T, axis=-1)@v, final_out, rtol=1e-4)


# --------------------------------- Pure matmul viz ------------------:
def matmul_viz():
    TPB = 3
    left_dim = 10
    right_dim = 10
    out = np.zeros((left_dim, right_dim))
    inner_dim = 5

    transpose_a = True
    transpose_b = False
    a = np.arange(left_dim * inner_dim).reshape(
        (left_dim, inner_dim) if not transpose_a else (inner_dim, left_dim)
    )
    b = np.arange(inner_dim * right_dim).reshape(
        (inner_dim, right_dim) if not transpose_b else (right_dim, inner_dim)
    )

    bpg_x = (left_dim + TPB - 1) // TPB
    bpg_y = (right_dim + TPB - 1) // TPB
    problem = CudaProblem(
        "Matrix multiplication",
        partial(matmul_kernel_factory, tpb=TPB, transpose_a=transpose_a, transpose_b=transpose_b),
        [a, b],
        out,
        blockspergrid=Coord(bpg_x, bpg_y),
        threadsperblock=Coord(TPB, TPB),
        spec=None,
    )
    problem.show(sparse=True, svg_height_factor=5)

#----------------------------------Plotting row-wise sum------------------:
def rowwise_sum_viz():
    TPB = 3
    ROW_SIZE = 10
    COL_SIZE = 5
    m = np.arange(ROW_SIZE * COL_SIZE).reshape(
        (ROW_SIZE, COL_SIZE)
    )

    THREADSPERBLOCK = Coord(TPB, TPB)
    BPG_x = (ROW_SIZE + TPB - 1) // TPB
    BPG_y = (COL_SIZE + TPB - 1) // TPB
    BLOCKSPERGRID = Coord(BPG_x, BPG_y)

    out = np.zeros((ROW_SIZE, BLOCKSPERGRID.y))
    problem = CudaProblem(
        "Row-wise sum",
        partial(rowwise_sum_factory, tpb=TPB),
        [m],
        out,
        args=[ROW_SIZE, COL_SIZE],
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
   