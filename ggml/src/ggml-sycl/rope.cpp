#include "rope.hpp"
#include "convert.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml.h"

struct rope_corr_dims {
    float v[2];
};

struct mrope_sections {
    int v[4];
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / sycl::max(0.001f, high - low);
    return 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

template <bool forward>
static void rope_yarn(const float theta_extrap, const float freq_scale,
                      const rope_corr_dims corr_dims, const int64_t i0,
                      const float ext_factor, float mscale, float &cos_theta,
                      float &sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix =
            rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static void rope_norm(const T *x, D *dst, const int ne00, const int ne01,
                      const int ne02, const int s01, const int s02,
                      const int s03, const int s1, const int s2, const int s3,
                      const int n_dims, const int n_offs, const int32_t *pos,
                      const float freq_scale, const float ext_factor,
                      const float attn_factor, const rope_corr_dims corr_dims,
                      const float theta_scale, const float *freq_factors,
                      const int64_t *row_indices, const int set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int idst = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = i0 + i1 * s01 + i2 * s02 + i3 * s03;

    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto &store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            sycl::float2 v = sycl::float2(x0, x1);
            ggml_sycl_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<sycl::half, D>) {
            sycl::half2 v = sycl::half2(x0, x1);
            ggml_sycl_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    const float freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta,
                    x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static void rope_neox(const T *x, D *dst, const int ne00, const int ne01,
                      const int ne02, const int s01, const int s02,
                      const int s03, const int s1, const int s2, const int s3,
                      const int n_dims, const int n_offs, const int32_t *pos,
                      const float freq_scale, const float ext_factor,
                      const float attn_factor, const rope_corr_dims corr_dims,
                      const float theta_scale, const float *freq_factors,
                      const int64_t *row_indices, const int set_rows_stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_sycl_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);

    const float freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs / 2 + 0];
    const float x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = ggml_sycl_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs / 2 + n_dims / 2] = ggml_sycl_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T>
static void rope_multi(const T *x, T *dst, const int ne00, const int ne01,
                       const int ne02, const int s01, const int s02,
                       const int s03, const int s1, const int s2, const int s3,
                       const int n_dims, const int n_offs, const int32_t *pos,
                       const float freq_scale, const float ext_factor,
                       const float attn_factor, const rope_corr_dims corr_dims,
                       const float theta_scale, const float *freq_factors,
                       const mrope_sections sections, const bool is_imrope) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
        dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const int sect_dims =
        sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * dpct::pow(theta_scale, iw / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * dpct::pow(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * dpct::pow(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * dpct::pow(theta_scale, iw / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[iw / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, iw,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs / 2 + 0];
    const float x1 = x[ix + n_offs / 2 + n_dims / 2];

    dst[idst + n_offs / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_offs / 2 + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <bool forward, bool has_ff, typename T>
static void rope_vision(const T *x, T *dst, const int ne00, const int ne01,
                        const int ne02, const int s01, const int s02,
                        const int s03, const int s1, const int s2, const int s3,
                        const int n_dims, const int32_t *pos,
                        const float freq_scale, const float ext_factor,
                        const float attn_factor, const rope_corr_dims corr_dims,
                        const float theta_scale, const float *freq_factors,
                        const mrope_sections sections) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                        item_ct1.get_local_id(1));

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                        item_ct1.get_local_id(2);

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int idst = i0 / 2 + i1 * s1 + i2 * s2 + i3 * s3;
    const int ix = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base = pos[i2] * dpct::pow(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base = pos[i2 + ne02] * dpct::pow(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                       ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <bool forward, typename T, typename D>
static void
rope_norm_sycl(const T *x, D *dst, const int ne00, const int ne01,
               const int ne02, const int s01, const int s02, const int s03,
               const int s1, const int s2, const int s3, const int n_dims,
               const int n_offs, const int nr, const int32_t *pos, const float freq_scale,
               const float freq_base, const float ext_factor,
               const float attn_factor, const rope_corr_dims corr_dims,
               const float *freq_factors, const int64_t *row_indices,
               const int set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, false>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_norm<forward, true>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <bool forward, typename T, typename D>
static void
rope_neox_sycl(const T *x, D *dst, const int ne00, const int ne01,
               const int ne02, const int s01, const int s02, const int s03,
               const int s1, const int s2, const int s3, const int n_dims,
               const int n_offs, const int nr, const int32_t *pos, const float freq_scale,
               const float freq_base, const float ext_factor,
               const float attn_factor, const rope_corr_dims corr_dims,
               const float *freq_factors, const int64_t *row_indices,
               const int set_rows_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, false>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_neox<forward, true>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, row_indices, set_rows_stride);
            });
    }
}

template <bool forward, typename T>
static void
rope_multi_sycl(const T *x, T *dst, const int ne00, const int ne01,
                const int ne02, const int s01, const int s02, const int s03,
                const int s1, const int s2, const int s3, const int n_dims,
                const int n_offs, const int nr, const int32_t *pos, const float freq_scale,
                const float freq_base, const float ext_factor,
                const float attn_factor, const rope_corr_dims corr_dims,
                const float *freq_factors, const mrope_sections sections,
                const bool is_imrope, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, false, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_multi<forward, true, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    n_offs, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections, is_imrope);
            });
    }
}

template <bool forward, typename T>
static void
rope_vision_sycl(const T *x, T *dst, const int ne00, const int ne01,
                 const int ne02, const int s01, const int s02, const int s03,
                 const int s1, const int s2, const int s3, const int n_dims,
                 const int nr, const int32_t *pos, const float freq_scale,
                 const float freq_base, const float ext_factor,
                 const float attn_factor, const rope_corr_dims corr_dims,
                 const float *freq_factors, const mrope_sections sections,
                 dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dpct::dim3 block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (ne00 + 2 * SYCL_ROPE_BLOCK_SIZE - 1) / (2 * SYCL_ROPE_BLOCK_SIZE);
    const dpct::dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, false, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                GGML_UNUSED(item_ct1);
                rope_vision<forward, true, T>(
                    x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                    pos, freq_scale, ext_factor, attn_factor, corr_dims,
                    theta_scale, freq_factors, sections);
            });
    }
}

template <bool forward>
void ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
                            const ggml_tensor *set_rows = nullptr) {
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    const ggml_tensor *src2 = dst->src[2];

    const float *src0_d = (const float *)src0->data;
    const float *src1_d = (const float *)src1->data;

    void *dst_d = dst->data;
    const int64_t *row_indices = nullptr;
    ggml_type dst_type = dst->type;
    int set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d = set_rows->data;
        row_indices = (const int64_t *)set_rows->src[1]->data;
        dst_type = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type ||
                (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    const int n_dims = ((int32_t *)dst->op_params)[1];
    const int mode = ((int32_t *)dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *)dst->op_params)[4];
    const int n_offs = ((int32_t *)dst->op_params)[15];
    mrope_sections sections;

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base, (int32_t *)dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, sizeof(float));
    memcpy(&sections.v, (int32_t *)dst->op_params + 11, sizeof(int) * 4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 ||
                    sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t *pos = (const int32_t *)src1_d;

    const float *freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *)src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_sycl<forward, float, float>(
                (const float *)src0_d, (float *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, float, sycl::half>(
                (const float *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_sycl<forward, sycl::half, sycl::half>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } else if (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_sycl<forward>((const float *)src0_d, (float *)dst_d,
                                     ne00, ne01, ne02, s01, s02, s03, s1, s2,
                                     s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                     ext_factor, attn_factor, corr_dims,
                                     freq_factors, sections, is_imrope, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_sycl<forward>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, is_imrope, stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } else if (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_sycl<forward>(
                (const float *)src0_d, (float *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, sections,
                stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_sycl<forward>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                sections, stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } else {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_sycl<forward, float, float>(
                (const float *)src0_d, (float *)dst_d, ne00, ne01, ne02, s01,
                s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, float, sycl::half>(
                (const float *)src0_d, (sycl::half *)dst_d, ne00, ne01, ne02,
                s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_sycl<forward, sycl::half, sycl::half>(
                (const sycl::half *)src0_d, (sycl::half *)dst_d, ne00, ne01,
                ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                row_indices, set_rows_stride, stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    }
}

void ggml_sycl_rope(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ggml_sycl_op_rope_impl<true>(ctx, dst);
}

void ggml_sycl_rope_back(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope_impl<false>(ctx, dst);
}

void ggml_sycl_rope_fused(ggml_backend_sycl_context &ctx, ggml_tensor *rope,
                          ggml_tensor *set_rows) {
    scope_op_debug_print scope_dbg_print(__func__, rope, /*num_src=*/3);
    ggml_sycl_op_rope_impl<true>(ctx, rope, set_rows);
}

void ggml_sycl_ornith_rms_mul_rope_qk_fused(
        ggml_backend_sycl_context & ctx,
        ggml_tensor * q_rms, ggml_tensor * q_mul, ggml_tensor * q_rope,
        ggml_tensor * k_rms, ggml_tensor * k_mul, ggml_tensor * k_rope,
        ggml_tensor * k_set_rows) {
    GGML_ASSERT(q_rms != nullptr && q_mul != nullptr && q_rope != nullptr);
    GGML_ASSERT(k_rms != nullptr && k_mul != nullptr && k_rope != nullptr && k_set_rows != nullptr);
    GGML_ASSERT(q_rms->op == GGML_OP_RMS_NORM && q_mul->op == GGML_OP_MUL && q_rope->op == GGML_OP_ROPE);
    GGML_ASSERT(k_rms->op == GGML_OP_RMS_NORM && k_mul->op == GGML_OP_MUL && k_rope->op == GGML_OP_ROPE);
    GGML_ASSERT(q_rope->src[0] == q_mul && k_rope->src[0] == k_mul);
    GGML_ASSERT((q_mul->src[0] == q_rms) != (q_mul->src[1] == q_rms));
    GGML_ASSERT((k_mul->src[0] == k_rms) != (k_mul->src[1] == k_rms));

    const ggml_tensor * q_src_tensor = q_rms->src[0];
    const ggml_tensor * k_src_tensor = k_rms->src[0];
    const ggml_tensor * q_weight_tensor = q_mul->src[0] == q_rms ? q_mul->src[1] : q_mul->src[0];
    const ggml_tensor * k_weight_tensor = k_mul->src[0] == k_rms ? k_mul->src[1] : k_mul->src[0];
    GGML_ASSERT(q_src_tensor != nullptr && k_src_tensor != nullptr);
    GGML_ASSERT(q_weight_tensor != nullptr && k_weight_tensor != nullptr);
    GGML_ASSERT(q_src_tensor->type == GGML_TYPE_F32 && k_src_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(q_weight_tensor->type == GGML_TYPE_F32 && k_weight_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(q_rope->type == GGML_TYPE_F32 && k_rope->type == GGML_TYPE_F32);
    GGML_ASSERT(q_src_tensor->ne[0] == 256 && q_src_tensor->ne[1] == 16 &&
                q_src_tensor->ne[2] == 1 && q_src_tensor->ne[3] == 1);
    GGML_ASSERT(k_src_tensor->ne[0] == 256 && k_src_tensor->ne[1] == 2 &&
                k_src_tensor->ne[2] == 1 && k_src_tensor->ne[3] == 1);
    GGML_ASSERT(q_weight_tensor->ne[0] == 256 && ggml_nelements(q_weight_tensor) == 256);
    GGML_ASSERT(k_weight_tensor->ne[0] == 256 && ggml_nelements(k_weight_tensor) == 256);
    GGML_ASSERT(q_src_tensor->nb[0] == sizeof(float) && k_src_tensor->nb[0] == sizeof(float));
    GGML_ASSERT(q_weight_tensor->nb[0] == sizeof(float) && k_weight_tensor->nb[0] == sizeof(float));
    GGML_ASSERT(q_rope->nb[0] == sizeof(float));
    GGML_ASSERT(k_set_rows->type == GGML_TYPE_F16 && k_set_rows->src[1] != nullptr &&
                k_set_rows->src[1]->type == GGML_TYPE_I64);
    GGML_ASSERT(q_rope->src[1] != nullptr && q_rope->src[1] == k_rope->src[1] &&
                q_rope->src[1]->type == GGML_TYPE_I32);

    float q_eps = 0.0f;
    float k_eps = 0.0f;
    memcpy(&q_eps, q_rms->op_params, sizeof(float));
    memcpy(&k_eps, k_rms->op_params, sizeof(float));
    GGML_ASSERT(q_eps >= 0.0f && q_eps == k_eps);

    const int32_t * ip = (const int32_t *) q_rope->op_params;
    GGML_ASSERT(memcmp(q_rope->op_params, k_rope->op_params, GGML_MAX_OP_PARAMS) == 0);
    GGML_ASSERT(ip[1] == 64 && ip[2] == GGML_ROPE_TYPE_IMROPE && ip[4] == 262144);

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base,   ip + 5,  sizeof(float));
    memcpy(&freq_scale,  ip + 6,  sizeof(float));
    memcpy(&ext_factor,  ip + 7,  sizeof(float));
    memcpy(&attn_factor, ip + 8,  sizeof(float));
    memcpy(&beta_fast,   ip + 9,  sizeof(float));
    memcpy(&beta_slow,   ip + 10, sizeof(float));
    GGML_ASSERT(freq_base == 10000000.0f && freq_scale == 1.0f &&
                ext_factor == 0.0f && attn_factor == 1.0f);

    mrope_sections sections;
    memcpy(&sections.v, ip + 11, sizeof(sections.v));
    GGML_ASSERT(sections.v[0] == 11 && sections.v[1] == 11 &&
                sections.v[2] == 10 && sections.v[3] == 0);

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(ip[1], ip[4], freq_base, beta_fast, beta_slow, corr_dims.v);
    const float theta_scale = powf(freq_base, -2.0f / ip[1]);

    const float * q_src = (const float *) q_src_tensor->data;
    const float * k_src = (const float *) k_src_tensor->data;
    const float * q_weight = (const float *) q_weight_tensor->data;
    const float * k_weight = (const float *) k_weight_tensor->data;
    float * q_dst = (float *) q_rope->data;
    sycl::half * k_dst = (sycl::half *) k_set_rows->data;
    const int32_t * pos = (const int32_t *) q_rope->src[1]->data;
    const int64_t * row_indices = (const int64_t *) k_set_rows->src[1]->data;

    const int q_s01 = q_src_tensor->nb[1] / sizeof(float);
    const int k_s01 = k_src_tensor->nb[1] / sizeof(float);
    const int q_d01 = q_rope->nb[1] / sizeof(float);
    const int k_set_rows_stride = k_set_rows->nb[1] / sizeof(sycl::half);
    static const bool poison_q = [] {
        const char * value = getenv("GGML_SYCL_FUSED_ORNITH_QK_NORM_ROPE_POISON");
        return value != nullptr && atoi(value) != 0;
    }();
    const bool poison_q_kernel = poison_q;

    constexpr int q_rows = 16;
    constexpr int k_rows = 2;
    constexpr int rms_lanes = WARP_SIZE;
    constexpr int rope_pairs = 128;
    const dpct::dim3 block_dims(1, rms_lanes, 1);
    const dpct::dim3 block_nums(q_rows + k_rows, 1, 1);
    dpct::queue_ptr stream = ctx.stream();
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> rms_mul_acc(sycl::range<1>(256), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item_ct1.get_group(2);
                const int tid = item_ct1.get_local_id(1);
                const bool is_q = row < q_rows;
                const int local_row = is_q ? row : row - q_rows;
                const float * src = is_q ? q_src + local_row * q_s01 : k_src + local_row * k_s01;
                const float * weight = is_q ? q_weight : k_weight;

                float sum = 0.0f;
                for (int col = tid; col < 256; col += rms_lanes) {
                    const float value = src[col];
                    sum += value * value;
                }
                sum = warp_reduce_sum(sum, item_ct1);
                const float scale = sycl::rsqrt(sum / 256.0f + q_eps);

                for (int col = tid; col < 256; col += rms_lanes) {
                    rms_mul_acc[col] = scale * src[col] * weight[col];
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);

                for (int pair = tid; pair < rope_pairs; pair += rms_lanes) {
                    const int i0 = 2 * pair;
                    int col0;
                    int col1;
                    int out0;
                    int out1;
                    if (i0 < 64) {
                        col0 = i0 / 2;
                        col1 = col0 + 32;
                        out0 = col0;
                        out1 = col1;
                    } else {
                        col0 = i0;
                        col1 = i0 + 1;
                        out0 = col0;
                        out1 = col1;
                    }

                    const float x0 = rms_mul_acc[col0];
                    const float x1 = rms_mul_acc[col1];
                    float y0 = x0;
                    float y1 = x1;

                    if (i0 < 64) {
                        const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
                        const int sector = (i0 / 2) % sect_dims;
                        float theta_base;
                        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {
                            theta_base = pos[1] * dpct::pow(theta_scale, i0 / 2.0f);
                        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {
                            theta_base = pos[2] * dpct::pow(theta_scale, i0 / 2.0f);
                        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {
                            theta_base = pos[0] * dpct::pow(theta_scale, i0 / 2.0f);
                        } else {
                            theta_base = pos[3] * dpct::pow(theta_scale, i0 / 2.0f);
                        }
                        float cos_theta;
                        float sin_theta;
                        rope_yarn<true>(theta_base, freq_scale, corr_dims, i0,
                                        ext_factor, attn_factor, cos_theta, sin_theta);
                        y0 = x0 * cos_theta - x1 * sin_theta;
                        y1 = x0 * sin_theta + x1 * cos_theta;
                    }

                    if (is_q) {
                        q_dst[local_row * q_d01 + out0] = poison_q_kernel ? 0.0f : y0;
                        q_dst[local_row * q_d01 + out1] = poison_q_kernel ? 0.0f : y1;
                    } else {
                        const int base = row_indices[0] * k_set_rows_stride + local_row * 256;
                        k_dst[base + out0] = ggml_sycl_cast<sycl::half>(y0);
                        k_dst[base + out1] = ggml_sycl_cast<sycl::half>(y1);
                    }
                }
            });
    });
}
