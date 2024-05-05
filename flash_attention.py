from numba import cuda
import numba
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)


import os
from numba import cuda
import numba
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)


def flash_attn_forward_no_stabilisation_kernel_factory(cuda, tpb_x, hidden_dim):
    """
    Assumptions: 
    1) The kernel will be launched on a block_size (tpb_x, hidden_dim) on a grid size (?, 1)
    2) tpb_x < hidden_dim, so we have enough threads to compute the (tpb_x, tpb_x) matrix
    3) tpb_x determines the parallelisation over seqlen, so we can reduce it as much as we want, to the detriment of speed.
    """
    def flash_attn_forward_no_stabilisation_kernel(out, q, k, v):
        # this will be launched as a tpb_x x hidden_dim block with hidden_dim > tpb_x
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        i = cuda.blockIdx.x * cuda.blockDim.x + local_i
        j = cuda.blockIdx.y * cuda.blockDim.y + local_j
        seqlen = q.shape[0]

        # we assume that hidden_dim is small enough that this can be loaded
        # into shared memory (we can reduce tpb_x, if not)
        q_shared = cuda.shared.array((tpb_x, hidden_dim), numba.float32)
        k_shared = cuda.shared.array((tpb_x, hidden_dim), numba.float32)
        v_shared = cuda.shared.array((tpb_x, hidden_dim), numba.float32)

        # output initialised as 0
        out_ij = 0.0

        exp_qkT = cuda.shared.array((tpb_x, tpb_x), numba.float32)
        rowsumexp_qkT = cuda.shared.array((tpb_x, tpb_x), numba.float32) # extra tpb_x for parallel scan
        rowsumexp_qkT[local_i, local_j] = 0.0
        rowsumexp_for_my_row_up_to_now = 0.0

        # populate the queries
        if i < seqlen and j < hidden_dim:
            q_shared[local_i, local_j] = q[i, j]
        cuda.syncthreads()

        num_tiles = (seqlen + tpb_x - 1) // tpb_x
        for tile in range(num_tiles):
            n_keys = tpb_x if tile != num_tiles - 1 else seqlen - (num_tiles - 1) * tpb_x
            # load the next tile of keys and values
            row_to_read = local_i + tile * tpb_x
            if row_to_read < seqlen and j < hidden_dim:
                k_shared[local_i, local_j] = k[row_to_read, j]
                v_shared[local_i, local_j] = v[row_to_read, j]
            cuda.syncthreads()

            # compute this tile of q@k.T
            if i < seqlen and local_j < n_keys:
                s_ij = 0.0
                for inner_idx in range(hidden_dim):
                    s_ij += q_shared[local_i, inner_idx] * k_shared[local_j, inner_idx]
                exped_s_ij = math.exp(s_ij)
                exp_qkT[local_i, local_j] = exped_s_ij
                rowsumexp_qkT[local_i, local_j] = exped_s_ij
            cuda.syncthreads()

            # compute the rowsum in the n_queries x n_keys block with parallel scan
            q = n_keys
            power_of_two = 1
            while q:
                if not local_j % (2 * power_of_two):
                    if i < seqlen and local_j + power_of_two < n_keys:
                        rowsumexp_qkT[local_i, local_j] = rowsumexp_qkT[local_i, local_j] + rowsumexp_qkT[local_i, local_j + power_of_two]
                power_of_two *= 2
                q = q // 2
                cuda.syncthreads()
            
            # update the rowsumexp
            rowsumexp_for_my_row_up_to_now += rowsumexp_qkT[local_i, 0]

            # compute exp_qkT @ V, assuming tpb_y >= hidden_dim
            if i < seqlen and local_j < hidden_dim:
                for inner_idx in range(n_keys):
                    out_ij += exp_qkT[local_i, inner_idx] * v_shared[inner_idx, local_j]

        cuda.syncthreads()
        if i < seqlen and j < hidden_dim:
            out[i, j] = out_ij / rowsumexp_for_my_row_up_to_now

    return flash_attn_forward_no_stabilisation_kernel


            

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def test_flash_attn_without_stabilisation():
    """In flash_attn, we assume that TPB_y >= hidden dim"""

    SEQ_LEN = 5 # weirdly fails at seq_len = 14
    HIDDEN_DIM = 4
    TPB_x = 3
    divider = 10
    np.random.seed(78)
    q = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider
    k = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider
    v = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider

    # k = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider
    # v = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider
    # q = np.arange(SEQ_LEN * HIDDEN_DIM).reshape(SEQ_LEN, HIDDEN_DIM).astype(np.float32) / divider

    flash_attn_forward_no_stabilisation_kernel = flash_attn_forward_no_stabilisation_kernel_factory(
        cuda, tpb_x=TPB_x, hidden_dim=HIDDEN_DIM
    )

    final_out = np.zeros((SEQ_LEN, HIDDEN_DIM))

    jitted_kernel = numba.cuda.jit(flash_attn_forward_no_stabilisation_kernel)[
        ((SEQ_LEN + TPB_x - 1) // TPB_x, 1),
        (TPB_x, HIDDEN_DIM)
    ]
    
    jitted_kernel(final_out, q, k, v)

    np.testing.assert_allclose(softmax(q@k.T, axis=-1)@v, final_out, rtol=1e-4)


if __name__ == "__main__":
    test_flash_attn_without_stabilisation()

