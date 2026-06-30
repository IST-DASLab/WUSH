/*
 * Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <common.h>

int backward_qt_bf16_cuda(
    const void*,
    const void*,
    const void*,
    const void*,
    void*,
    void*,
    const int,
    const int,
    const int,
    cudaStream_t
);

int backward_t_bf16_cuda(
    const void*,
    const void*,
    void*,
    void*,
    const int,
    const int,
    const int,
    cudaStream_t
);

int backward_bf16_square_double_mxfp8_cuda(
    const void*,
    const int,
    const int,
    void*,
    void*,
    void*,
    cudaStream_t
);

int mxfp4_transpose_mxfp8_cuda(
    const void*,
    const void*,
    const int,
    const int,
    void*,
    void*,
    cudaStream_t
);