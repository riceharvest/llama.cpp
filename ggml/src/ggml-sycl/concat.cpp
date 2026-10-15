//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "ggml.h"

#include "concat.hpp"

static inline size_t elem_size(ggml_type t) {
    return ggml_type_size(t) / ggml_blck_size(t);
}

template <typename T>
static void concat_T_dim0(const T *x, const T *y, T *dst,
                            const int ne0, const int ne00,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (nidx < ne00) { // src0
    int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                     item_ct1.get_group(0) * ne00 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx - ne00 + item_ct1.get_group(1) * (ne0 - ne00) +
        item_ct1.get_group(0) * (ne0 - ne00) * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
static void concat_T_dim1(const T *x, const T *y, T *dst,
                            const int ne0, const int ne01,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (item_ct1.get_group(1) < (size_t) ne01) { // src0
    int offset_src =
        nidx + item_ct1.get_group(1) * ne0 + item_ct1.get_group(0) * ne0 * ne01;
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx + (item_ct1.get_group(1) - ne01) * ne0 +
        item_ct1.get_group(0) * ne0 * (item_ct1.get_group_range(1) - ne01);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
static void concat_T_dim2(const T *x, const T *y, T *dst,
                            const int ne0, const int ne02,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (item_ct1.get_group(0) < (size_t) ne02) { // src0
    int offset_src = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx + item_ct1.get_group(1) * ne0 +
        (item_ct1.get_group(0) - ne02) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
static void concat_T_sycl(const T *x, const T *y, T *dst,
                            int ne00, int ne01, int ne02, int ne0, int ne1,
                            int ne2, int dim, queue_ptr stream) {
  int num_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> gridDim(ne2, ne1, num_blocks);
  switch (dim) {
  case 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, item_ct1); });
      break;
  case 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, item_ct1); });
      break;
  // dim >=2 will be dispatched to the default path
  default:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, item_ct1); });
      break;
  }
}

// non-contiguous kernel (slow)
template<typename T>
static void concat_T_sycl_non_cont(
    queue_ptr stream, const char *src0, const char *src1, char *dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03, uint64_t nb00,
    uint64_t nb01, uint64_t nb02, uint64_t nb03, int64_t /*ne10*/,
    int64_t /*ne11*/, int64_t /*ne12*/, int64_t /*ne13*/, uint64_t nb10,
    uint64_t nb11, uint64_t nb12, uint64_t nb13, int64_t ne0, int64_t ne1,
    int64_t ne2, int64_t ne3, uint64_t nb0, uint64_t nb1, uint64_t nb2,
    uint64_t nb3, int32_t dim) {
  sycl::range<3> gridDim(ne3, ne2, ne1);

  // Avoid oversubscribing device when there is not enough elements along the innermost dim to
  // fill a full SYCL_CONCAT_BLOCK_SIZE. For larger # of elements, the full SYCL_CONCAT_BLOCK_SIZE
  // is used.
  const int64_t ne0_pad   = GGML_PAD(ne0, WARP_SIZE);
  const int64_t block_ne0 = ne0_pad < SYCL_CONCAT_BLOCK_SIZE ? ne0_pad : (int64_t) SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> blockDim(1, 1, block_ne0);

  stream->parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
      int64_t i3 = item_ct1.get_group(0);
      int64_t i2 = item_ct1.get_group(1);
      int64_t i1 = item_ct1.get_group(2);

      int64_t o[4] = { 0, 0, 0, 0 };
      o[dim]       = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

      const T * x;

      for (int i0 = item_ct1.get_local_id(2); i0 < ne0; i0 += item_ct1.get_local_range(2)) {
          if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (const T *) (src0 + (i3) *nb03 + (i2) *nb02 + (i1) *nb01 + (i0) *nb00);
          } else {
              x = (const T *) (src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 + (i1 - o[1]) * nb11 +
                                   (i0 - o[0]) * nb10);
          }

          T *y = (T *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
      }
  });
}

template <typename T>
void concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *  src0   = dst->src[0];
    const ggml_tensor *  src1   = dst->src[1];
    queue_ptr            stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T * dst_d = (T *) dst->data;
        size_t type_size = elem_size(dst->type);
        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_size), src1_d + i3 * (src1->nb[3] / type_size),
                                dst_d + i3 * (dst->nb[3] / type_size), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d + size0 / type_size, src1_d, size1)));
        }
    } else {
        concat_T_sycl_non_cont<T>(stream, (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

static void concat_impl_q4_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0   = dst->src[0];
    const ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(src0->ne[0] % QK4_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_0 == 0);

    const int ne00_blk = src0->ne[0] / QK4_0;
    const int ne0_blk  = dst->ne[0] / QK4_0;

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const block_q4_0 * src0_d = (const block_q4_0 *) src0->data;
        const block_q4_0 * src1_d = (const block_q4_0 *) src1->data;
        block_q4_0 * dst_d        = (block_q4_0 *) dst->data;
        const size_t type_size    = sizeof(block_q4_0);

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q4_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((char *) dst_d + size0, src1_d, size1)));
        }
    } else {
        concat_T_sycl_non_cont<block_q4_0>(
            stream, (const char *) src0->data, (const char *) src1->data,
            (char *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

static void concat_impl_q4_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0   = dst->src[0];
    const ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(src0->ne[0] % QK4_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK4_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK4_1 == 0);

    const int ne00_blk = src0->ne[0] / QK4_1;
    const int ne0_blk  = dst->ne[0] / QK4_1;

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const block_q4_1 * src0_d = (const block_q4_1 *) src0->data;
        const block_q4_1 * src1_d = (const block_q4_1 *) src1->data;
        block_q4_1 * dst_d        = (block_q4_1 *) dst->data;
        const size_t type_size    = sizeof(block_q4_1);

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q4_1>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((char *) dst_d + size0, src1_d, size1)));
        }
    } else {
        concat_T_sycl_non_cont<block_q4_1>(
            stream, (const char *) src0->data, (const char *) src1->data,
            (char *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK4_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

static void concat_impl_q5_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0   = dst->src[0];
    const ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(src0->ne[0] % QK5_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_0 == 0);

    const int ne00_blk = src0->ne[0] / QK5_0;
    const int ne0_blk  = dst->ne[0] / QK5_0;

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const block_q5_0 * src0_d = (const block_q5_0 *) src0->data;
        const block_q5_0 * src1_d = (const block_q5_0 *) src1->data;
        block_q5_0 * dst_d        = (block_q5_0 *) dst->data;
        const size_t type_size    = sizeof(block_q5_0);

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q5_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((char *) dst_d + size0, src1_d, size1)));
        }
    } else {
        concat_T_sycl_non_cont<block_q5_0>(
            stream, (const char *) src0->data, (const char *) src1->data,
            (char *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

static void concat_impl_q5_1_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0   = dst->src[0];
    const ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src1->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(dst->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(src0->ne[0] % QK5_1 == 0);
    GGML_ASSERT(src1->ne[0] % QK5_1 == 0);
    GGML_ASSERT(dst->ne[0] % QK5_1 == 0);

    const int ne00_blk = src0->ne[0] / QK5_1;
    const int ne0_blk  = dst->ne[0] / QK5_1;

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const block_q5_1 * src0_d = (const block_q5_1 *) src0->data;
        const block_q5_1 * src1_d = (const block_q5_1 *) src1->data;
        block_q5_1 * dst_d        = (block_q5_1 *) dst->data;
        const size_t type_size    = sizeof(block_q5_1);

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q5_1>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((char *) dst_d + size0, src1_d, size1)));
        }
    } else {
        concat_T_sycl_non_cont<block_q5_1>(
            stream, (const char *) src0->data, (const char *) src1->data,
            (char *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK5_1, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

static void concat_impl_q8_0_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0   = dst->src[0];
    const ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src0->ne[0] % QK8_0 == 0);
    GGML_ASSERT(src1->ne[0] % QK8_0 == 0);
    GGML_ASSERT(dst->ne[0] % QK8_0 == 0);

    const int ne00_blk = src0->ne[0] / QK8_0;
    const int ne0_blk  = dst->ne[0] / QK8_0;

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const block_q8_0 * src0_d = (const block_q8_0 *) src0->data;
        const block_q8_0 * src1_d = (const block_q8_0 *) src1->data;
        block_q8_0 * dst_d        = (block_q8_0 *) dst->data;
        const size_t type_size    = sizeof(block_q8_0);

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_q8_0>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((char *) dst_d + size0, src1_d, size1)));
        }
    } else {
        concat_T_sycl_non_cont<block_q8_0>(
            stream, (const char *) src0->data, (const char *) src1->data,
            (char *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / QK8_0, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    switch (dst->type) {
    case GGML_TYPE_F32:
        concat_impl_sycl<float>(ctx, dst);
        break;
    case GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        break;
#ifdef GGML_SYCL_HAS_BF16
    case GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        break;
#endif
    case GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        break;
    case GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        break;
    case GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        break;
    case GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        break;
    case GGML_TYPE_Q4_0:
        concat_impl_q4_0_sycl(ctx, dst);
        break;
    case GGML_TYPE_Q4_1:
        concat_impl_q4_1_sycl(ctx, dst);
        break;
    case GGML_TYPE_Q5_0:
        concat_impl_q5_0_sycl(ctx, dst);
        break;
    case GGML_TYPE_Q5_1:
        concat_impl_q5_1_sycl(ctx, dst);
        break;
    case GGML_TYPE_Q8_0:
        concat_impl_q8_0_sycl(ctx, dst);
        break;
    default:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(false);
    break;
    }
}

void ggml_sycl_concat_state_update(
        ggml_backend_sycl_context & ctx, ggml_tensor * concat, ggml_tensor * cpy) {
    GGML_ASSERT(concat != nullptr && cpy != nullptr);
    GGML_ASSERT(concat->op == GGML_OP_CONCAT && cpy->op == GGML_OP_CPY);
    GGML_ASSERT(concat->src[0] != nullptr && concat->src[1] != nullptr);
    GGML_ASSERT(cpy->src[0] != nullptr && cpy->src[1] != nullptr);
    GGML_ASSERT(concat->type == GGML_TYPE_F32);
    GGML_ASSERT(concat->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(concat->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(cpy->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(cpy->src[1]->type == GGML_TYPE_F32);

    const int32_t dim = ((const int32_t *) concat->op_params)[0];
    GGML_ASSERT(dim == 0);
    GGML_ASSERT(concat->src[0]->ne[0] + concat->src[1]->ne[0] == concat->ne[0]);
    GGML_ASSERT(concat->src[1]->ne[0] == 1);
    GGML_ASSERT(cpy->src[0]->ne[0] == concat->ne[0] - 1);
    GGML_ASSERT(ggml_nelements(cpy->src[0]) == ggml_nelements(cpy->src[1]));
    GGML_ASSERT(ggml_is_contiguous(concat));
    GGML_ASSERT(ggml_is_contiguous(cpy->src[1]));
    GGML_ASSERT(cpy->src[0]->data == (char *) concat->data + concat->nb[0]);
    GGML_ASSERT(cpy->src[0]->nb[0] == concat->nb[0]);
    GGML_ASSERT(cpy->src[0]->nb[1] == concat->nb[1]);
    GGML_ASSERT(cpy->src[0]->nb[2] == concat->nb[2]);
    GGML_ASSERT(cpy->src[0]->nb[3] == concat->nb[3]);

    const ggml_tensor * src0 = concat->src[0];
    const ggml_tensor * src1 = concat->src[1];
    ggml_tensor * state = cpy->src[1];

    const char * src0_data = (const char *) src0->data;
    const char * src1_data = (const char *) src1->data;
    char * concat_data = (char *) concat->data;
    float * state_data = (float *) state->data;

    const int64_t ne0 = concat->ne[0];
    const int64_t ne1 = concat->ne[1];
    const int64_t ne2 = concat->ne[2];
    const int64_t ne3 = concat->ne[3];
    const int64_t ne00 = src0->ne[0];
    const uint64_t nb00 = src0->nb[0];
    const uint64_t nb01 = src0->nb[1];
    const uint64_t nb02 = src0->nb[2];
    const uint64_t nb03 = src0->nb[3];
    const uint64_t nb10 = src1->nb[0];
    const uint64_t nb11 = src1->nb[1];
    const uint64_t nb12 = src1->nb[2];
    const uint64_t nb13 = src1->nb[3];
    const uint64_t nb0 = concat->nb[0];
    const uint64_t nb1 = concat->nb[1];
    const uint64_t nb2 = concat->nb[2];
    const uint64_t nb3 = concat->nb[3];
    const size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;
    constexpr size_t wg = 256;
    const size_t global = ((total + wg - 1) / wg) * wg;

    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> item) {
            const size_t linear = item.get_global_id(0);
            if (linear >= total) {
                return;
            }

            size_t rem = linear;
            const int64_t i0 = rem % ne0; rem /= ne0;
            const int64_t i1 = rem % ne1; rem /= ne1;
            const int64_t i2 = rem % ne2; rem /= ne2;
            const int64_t i3 = rem;

            const char * in = i0 < ne00
                ? src0_data + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03
                : src1_data + (i0 - ne00) * nb10 + i1 * nb11 + i2 * nb12 + i3 * nb13;
            const float value = *(const float *) in;
            *(float *) (concat_data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3) = value;

            if (i0 != 0) {
                const size_t state_index = (size_t) (i0 - 1) + (size_t) (ne0 - 1) *
                    ((size_t) i1 + (size_t) ne1 * ((size_t) i2 + (size_t) ne2 * (size_t) i3));
                state_data[state_index] = value;
            }
        });
}

void ggml_sycl_concat_state_update_direct(
        ggml_backend_sycl_context & ctx, ggml_tensor * get_rows,
        ggml_tensor * concat, ggml_tensor * cpy) {
    GGML_ASSERT(get_rows != nullptr && concat != nullptr && cpy != nullptr);
    GGML_ASSERT(get_rows->op == GGML_OP_GET_ROWS);
    GGML_ASSERT(concat->op == GGML_OP_CONCAT && cpy->op == GGML_OP_CPY);
    GGML_ASSERT(get_rows->src[0] != nullptr && get_rows->src[1] != nullptr);
    GGML_ASSERT(concat->src[0] != nullptr && concat->src[1] != nullptr);
    GGML_ASSERT(cpy->src[0] != nullptr && cpy->src[1] != nullptr);
    GGML_ASSERT(get_rows->type == GGML_TYPE_F32);
    GGML_ASSERT(get_rows->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(get_rows->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(concat->type == GGML_TYPE_F32);
    GGML_ASSERT(concat->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(concat->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(cpy->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(cpy->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(get_rows));
    GGML_ASSERT(ggml_is_contiguous(get_rows->src[0]));
    GGML_ASSERT(ggml_nrows(get_rows) == 1);
    GGML_ASSERT(ggml_nrows(get_rows->src[0]) == 1);
    GGML_ASSERT(ggml_nelements(get_rows) == 3 * 8192);
    GGML_ASSERT(ggml_nelements(get_rows->src[0]) == ggml_nelements(get_rows));
    GGML_ASSERT(concat->ne[0] == 4 && concat->ne[1] == 8192);
    GGML_ASSERT(concat->ne[2] == 1 && concat->ne[3] == 1);
    GGML_ASSERT(concat->src[0]->ne[0] == 3 && concat->src[0]->ne[1] == 8192);
    GGML_ASSERT(concat->src[1]->ne[0] == 1 && concat->src[1]->ne[1] == 8192);
    GGML_ASSERT(ggml_is_contiguous(concat));
    GGML_ASSERT(ggml_is_contiguous(cpy->src[1]));
    GGML_ASSERT(concat->src[0]->data == get_rows->data);
    GGML_ASSERT(cpy->src[1]->data == get_rows->src[0]->data);
    GGML_ASSERT(cpy->src[0]->data == (char *) concat->data + concat->nb[0]);

    const float * persistent_in = (const float *) get_rows->src[0]->data;
    float * get_rows_out = (float *) get_rows->data;
    const char * current = (const char *) concat->src[1]->data;
    char * concat_out = (char *) concat->data;
    float * persistent_out = (float *) cpy->src[1]->data;

    const uint64_t current_nb1 = concat->src[1]->nb[1];
    const uint64_t concat_nb0 = concat->nb[0];
    const uint64_t concat_nb1 = concat->nb[1];
    constexpr size_t channels = 8192;
    constexpr size_t wg = 256;

    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(channels), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> item) {
            const size_t channel = item.get_global_id(0);
            const size_t old_base = 3 * channel;

            // Load the complete old state before writing the overlapping persistent buffer.
            const float old0 = persistent_in[old_base + 0];
            const float old1 = persistent_in[old_base + 1];
            const float old2 = persistent_in[old_base + 2];
            const float value = *(const float *) (current + channel * current_nb1);

            // Materialize every graph-visible result even though their stock kernels are skipped.
            get_rows_out[old_base + 0] = old0;
            get_rows_out[old_base + 1] = old1;
            get_rows_out[old_base + 2] = old2;
            *(float *) (concat_out + channel * concat_nb1 + 0 * concat_nb0) = old0;
            *(float *) (concat_out + channel * concat_nb1 + 1 * concat_nb0) = old1;
            *(float *) (concat_out + channel * concat_nb1 + 2 * concat_nb0) = old2;
            *(float *) (concat_out + channel * concat_nb1 + 3 * concat_nb0) = value;

            persistent_out[old_base + 0] = old1;
            persistent_out[old_base + 1] = old2;
            persistent_out[old_base + 2] = value;
        });
}
