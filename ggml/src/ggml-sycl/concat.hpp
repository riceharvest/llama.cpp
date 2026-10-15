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

#ifndef GGML_SYCL_CONCAT_HPP
#define GGML_SYCL_CONCAT_HPP

#include "common.hpp"

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

void ggml_sycl_concat_state_update(
        ggml_backend_sycl_context & ctx, ggml_tensor * concat, ggml_tensor * cpy);

void ggml_sycl_concat_state_update_direct(
        ggml_backend_sycl_context & ctx, ggml_tensor * get_rows,
        ggml_tensor * concat, ggml_tensor * cpy);

#endif // GGML_SYCL_CONCAT_HPP
