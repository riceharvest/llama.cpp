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

#ifndef GGML_SYCL_NORM_HPP
#define GGML_SYCL_NORM_HPP

#include "common.hpp"

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

void ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

void ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor);

void ggml_sycl_op_ornith_gdn_rms_silu_gate_fused(
        ggml_backend_sycl_context& ctx, ggml_tensor* rms_norm,
        ggml_tensor* weight_mul, ggml_tensor* silu, ggml_tensor* gate_mul);

void ggml_sycl_op_residual_rms_norm_fused(
        ggml_backend_sycl_context& ctx, ggml_tensor* add, ggml_tensor* rms_norm, ggml_tensor* mul);

void ggml_sycl_op_moe_shared_residual_rms_norm_fused(
        ggml_backend_sycl_context& ctx, ggml_tensor* ffn_add,
        ggml_tensor* residual_add, ggml_tensor* rms_norm, ggml_tensor* mul);

void ggml_sycl_op_moe_shared_gate_residual_rms_norm_fused(
        ggml_backend_sycl_context& ctx, ggml_tensor* gate_sigmoid,
        ggml_tensor* shared_gated, ggml_tensor* ffn_add,
        ggml_tensor* residual_add, ggml_tensor* rms_norm, ggml_tensor* mul);

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#define GGML_SYCL_L2_BATCH_MAX 8
void ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, int count);

#endif // GGML_SYCL_NORM_HPP
