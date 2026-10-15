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

#ifndef GGML_SYCL_ROPE_HPP
#define GGML_SYCL_ROPE_HPP

#include "common.hpp"

void ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

void ggml_sycl_rope_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_rope_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// Ornith/Qwen3.5 one-token full attention: preserve the stock SIMD16 RMS
// reduction and FP32 RMS+weight materialization while joining Q/K IMRoPE and
// writing K directly to the F16 cache.
void ggml_sycl_ornith_rms_mul_rope_qk_fused(
        ggml_backend_sycl_context & ctx,
        ggml_tensor * q_rms, ggml_tensor * q_mul, ggml_tensor * q_rope,
        ggml_tensor * k_rms, ggml_tensor * k_mul, ggml_tensor * k_rope,
        ggml_tensor * k_set_rows);

#endif // GGML_SYCL_ROPE_HPP
