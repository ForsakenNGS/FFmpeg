/*
 * Copyright (c) 2017 Vittorio Giovara <vittorio.giovara@gmail.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * tonemap algorithms
 */

#include <float.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "colorspace.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

#include "tonemap.h"

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_GBRPF32,
    AV_PIX_FMT_GBRAPF32,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_NONE,
};

typedef struct ThreadData {
    AVFrame *in, *out;
    const AVPixFmtDescriptor *desc, *odesc;
    double peak;
} ThreadData;

static int query_formats(AVFilterContext *ctx)
{
    AVFilterFormats *formats = ff_make_format_list(pix_fmts);
    int res = ff_formats_ref(formats, &ctx->inputs[0]->out_formats);
    if (res < 0)
        return res;
    formats = NULL;
    res = ff_add_format(&formats, AV_PIX_FMT_NV12);
    if (res < 0)
        return res;
    return ff_formats_ref(formats, &ctx->outputs[0]->in_formats);
}

static av_cold int init(AVFilterContext *ctx)
{
    TonemapContext *s = ctx->priv;

    switch(s->tonemap) {
    case TONEMAP_GAMMA:
        if (isnan(s->param))
            s->param = 1.8f;
        break;
    case TONEMAP_REINHARD:
        if (!isnan(s->param))
            s->param = (1.0f - s->param) / s->param;
        break;
    case TONEMAP_MOBIUS:
        if (isnan(s->param))
            s->param = 0.3f;
        break;
    }

    if (isnan(s->param))
        s->param = 1.0f;

    s->tonemap_frame_p010_nv12 = ff_tonemap_frame_p010_nv12_c;

    return 0;
}

static float hable(float in)
{
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}

static float mobius(float in, float j, double peak)
{
    float a, b;

    if (in <= j)
        return in;

    a = -j * j * (peak - 1.0f) / (j * j - 2.0f * j + peak);
    b = (j * j - 2.0f * j * peak + peak) / FFMAX(peak - 1.0f, 1e-6);

    return (b * b + 2.0f * b * j + j * j) / (b - a) * (in + a) / (in + b);
}

static float eotf_st2084(float x) {
#define ST2084_MAX_LUMINANCE 10000.0f
#define REFERENCE_WHITE 100.0f
#define ST2084_M1 0.1593017578125f
#define ST2084_M2 78.84375f
#define ST2084_C1 0.8359375f
#define ST2084_C2 18.8515625f
#define ST2084_C3 18.6875f

    float p = powf(x, 1.0f / ST2084_M2);
    float a = FFMAX(p -ST2084_C1, 0.0f);
    float b = FFMAX(ST2084_C2 - ST2084_C3 * p, 1e-6f);
    float c  = powf(a / b, 1.0f / ST2084_M1);
    return x > 0.0f ? c * ST2084_MAX_LUMINANCE / REFERENCE_WHITE : 0.0f;
}

static float inverse_eotf_st2084(float x) {
    x *= REFERENCE_WHITE / ST2084_MAX_LUMINANCE;
    x = powf(x, ST2084_M1);
    x = (ST2084_C1 + ST2084_C2 * x) / (1.0f + ST2084_C3 * x);
    return powf(x, ST2084_M2);
}

static float bt2390(float sig_orig, double peak)
{
    float sig_pq = sig_orig / peak;
    float maxLum = 0.751829f / peak; // SDR peak in PQ

    float ks = 1.5f * maxLum - 0.5f;
    float tb = (sig_pq - ks) / (1.0f - ks);
    float tb2 = tb * tb;
    float tb3 = tb2 * tb;
    float pb = (2.0f * tb3 - 3.0f * tb2 + 1.0f) * ks +
               (tb3 - 2.0f * tb2 + tb) * (1.0f - ks) +
               (-2.0f * tb3 + 3.0f * tb2) * maxLum;
    float sig = (sig_pq < ks) ? sig_pq : pb;
    return eotf_st2084(sig * peak);
}

static float mapsig(enum TonemapAlgorithm alg, float sig, double peak, double param)
{
    switch(alg) {
    default:
    case TONEMAP_NONE:
        // do nothing
        break;
    case TONEMAP_LINEAR:
        sig = sig * param / peak;
        break;
    case TONEMAP_GAMMA:
        sig = sig > 0.05f ? pow(sig / peak, 1.0f / param)
                          : sig * pow(0.05f / peak, 1.0f / param) / 0.05f;
        break;
    case TONEMAP_CLIP:
        sig = av_clipf(sig * param, 0, 1.0f);
        break;
    case TONEMAP_HABLE:
        sig = hable(sig) / hable(peak);
        break;
    case TONEMAP_REINHARD:
        sig = sig / (sig + param) * (peak + param) / peak;
        break;
    case TONEMAP_MOBIUS:
        sig = mobius(sig, param, peak);
        break;
    case TONEMAP_BT2390:
        sig = bt2390(sig, peak);
        break;
    }

    return sig;
}

#define MIX(x,y,a) (x) * (1 - (a)) + (y) * (a)
static void tonemap(float r_in, float g_in, float b_in,
                    float *r_out, float *g_out, float *b_out,
                    double param, double desat, double peak,
                    const struct LumaCoefficients *coeffs,
                    enum TonemapAlgorithm alg)
{
    float sig, sig_orig;

    /* load values */
    *r_out = r_in;
    *g_out = g_in;
    *b_out = b_in;

    /* desaturate to prevent unnatural colors */
    if (desat > 0) {
        float luma = coeffs->cr * r_in + coeffs->cg * g_in + coeffs->cb * b_in;
        float overbright = FFMAX(luma - desat, 1e-6) / FFMAX(luma, 1e-6);
        *r_out = MIX(r_in, luma, overbright);
        *g_out = MIX(g_in, luma, overbright);
        *b_out = MIX(b_in, luma, overbright);
    }

    /* pick the brightest component, reducing the value range as necessary
     * to keep the entire signal in range and preventing discoloration due to
     * out-of-bounds clipping */
    sig = FFMAX(FFMAX3(*r_out, *g_out, *b_out), 1e-6);
    sig_orig = sig;

    sig = mapsig(alg, sig, peak, param);

    /* apply the computed scale factor to the color,
     * linearly to prevent discoloration */
    *r_out *= sig / sig_orig;
    *g_out *= sig / sig_orig;
    *b_out *= sig / sig_orig;
}

typedef struct ThreadData {
    AVFrame *in, *out;
    const AVPixFmtDescriptor *desc;
    double peak;
} ThreadData;

static int tonemap_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    TonemapContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in = td->in;
    AVFrame *out = td->out;
    const AVPixFmtDescriptor *desc = td->desc;
    const int slice_start = (in->height * jobnr) / nb_jobs;
    const int slice_end = (in->height * (jobnr+1)) / nb_jobs;
    double peak = td->peak;

    for (int y = slice_start; y < slice_end; y++)
        for (int x = 0; x < out->width; x++)
            tonemap(s, out, in, desc, x, y, peak);

    return 0;
}

static int filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext *ctx = link->dst;
    TonemapContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    ThreadData td;
    AVFrame *out;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(link->format);
    const AVPixFmtDescriptor *odesc = av_pix_fmt_desc_get(outlink->format);
    int ret;
    double peak = s->peak;
    const struct LumaCoefficients *coeffs = ff_get_luma_coefficients(in->colorspace);
    ThreadData td;

    if (!desc || !odesc) {
        av_frame_free(&in);
        return AVERROR_BUG;
    }

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    if ((ret = av_frame_copy_props(out, in)) < 0)
        goto fail;

    /* read peak from side data if not passed in */
    if (!peak) {
        peak = ff_determine_signal_peak(in);
        av_log(s, AV_LOG_DEBUG, "Computed signal peak: %f\n", peak);
    }

    /* input and output transfer will be linear */
    if (desc->flags & AV_PIX_FMT_FLAG_FLOAT) {
        if (in->color_trc == AVCOL_TRC_UNSPECIFIED) {
            av_log(s, AV_LOG_WARNING, "Untagged transfer, assuming linear light\n");
            out->color_trc = AVCOL_TRC_LINEAR;
        } else if (in->color_trc != AVCOL_TRC_LINEAR)
            av_log(s, AV_LOG_WARNING, "Tonemapping works on linear light only\n");
    } else {
        if (in->color_trc == AVCOL_TRC_UNSPECIFIED) {
            av_log(s, AV_LOG_WARNING, "Untagged transfer, assuming PQ\n");
            out->color_trc = AVCOL_TRC_SMPTEST2084;
        } else if (in->color_trc != AVCOL_TRC_SMPTEST2084)
            av_log(s, AV_LOG_WARNING, "Tonemapping works on PQ only\n");

        out->color_trc       = AVCOL_TRC_BT709;
        out->colorspace      = AVCOL_SPC_BT709;
        out->color_primaries = AVCOL_PRI_BT709;

        if (!s->lin_lut || !s->delin_lut) {
            if ((ret = comput_trc_luts(s, in->color_trc, out->color_trc)) < 0)
                goto fail;
        }

        if (!s->tonemap_lut || s->lut_peak != peak) {
            s->lut_peak = peak;
            if ((ret = compute_tonemap_lut(s)) < 0)
                goto fail;
        }

        if (s->coeffs != coeffs) {
            enum AVColorPrimaries iprm = in->color_primaries;
            s->ocoeffs = ff_get_luma_coefficients(out->colorspace);
            if ((ret = compute_yuv_coeffs(s, coeffs, s->ocoeffs, desc, odesc,
                 in->color_range, out->color_range)) < 0)
                goto fail;
            if (iprm == AVCOL_PRI_UNSPECIFIED)
                iprm = AVCOL_PRI_BT2020;
            if ((ret = compute_rgb_coeffs(s, iprm, out->color_primaries)) < 0)
                goto fail;
        }
    }

    /* load original color space even if pixel format is RGB to compute overbrights */
    s->coeffs = coeffs;
    if (s->desat > 0 && !s->coeffs) {
        if (in->colorspace == AVCOL_SPC_UNSPECIFIED)
            av_log(s, AV_LOG_WARNING, "Missing color space information, ");
        else if (!s->coeffs)
            av_log(s, AV_LOG_WARNING, "Unsupported color space '%s', ",
                   av_color_space_name(in->colorspace));
        av_log(s, AV_LOG_WARNING, "desaturation is disabled\n");
        s->desat = 0;
    }

    /* do the tone map */
    td.out = out;
    td.in = in;
    td.desc = desc;
    td.peak = peak;
    ctx->internal->execute(ctx, tonemap_slice, &td, NULL, FFMIN(in->height, ff_filter_get_nb_threads(ctx)));

    /* copy/generate alpha if needed */
    if (desc->flags & AV_PIX_FMT_FLAG_ALPHA && odesc->flags & AV_PIX_FMT_FLAG_ALPHA) {
        av_image_copy_plane(out->data[3], out->linesize[3],
                            in->data[3], in->linesize[3],
                            out->linesize[3], outlink->h);
    } else if (odesc->flags & AV_PIX_FMT_FLAG_ALPHA) {
        for (y = 0; y < out->height; y++) {
            for (x = 0; x < out->width; x++) {
                AV_WN32(out->data[3] + x * odesc->comp[3].step + y * out->linesize[3],
                        av_float2int(1.0f));
            }
        }
    }

    av_frame_free(&in);

    ff_update_hdr_metadata(out, peak);

    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static void uninit(AVFilterContext *ctx)
{
    TonemapContext *s = ctx->priv;

    av_freep(&s->lin_lut);
    av_freep(&s->delin_lut);
    av_freep(&s->tonemap_lut);
}

#define OFFSET(x) offsetof(TonemapContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
static const AVOption tonemap_options[] = {
    { "tonemap",      "tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, {.i64 = TONEMAP_BT2390}, TONEMAP_NONE, TONEMAP_MAX - 1, FLAGS, "tonemap" },
    {     "none",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_NONE},              0, 0, FLAGS, "tonemap" },
    {     "linear",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_LINEAR},            0, 0, FLAGS, "tonemap" },
    {     "gamma",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_GAMMA},             0, 0, FLAGS, "tonemap" },
    {     "clip",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_CLIP},              0, 0, FLAGS, "tonemap" },
    {     "reinhard", 0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_REINHARD},          0, 0, FLAGS, "tonemap" },
    {     "hable",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_HABLE},             0, 0, FLAGS, "tonemap" },
    {     "mobius",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_MOBIUS},            0, 0, FLAGS, "tonemap" },
    {     "bt2390",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_BT2390},            0, 0, FLAGS, "tonemap" },
    { "param",        "tonemap parameter", OFFSET(param), AV_OPT_TYPE_DOUBLE, {.dbl = NAN}, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",        "desaturation strength", OFFSET(desat), AV_OPT_TYPE_DOUBLE, {.dbl = 2}, 0, DBL_MAX, FLAGS },
    { "peak",         "signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, {.dbl = 0}, 0, DBL_MAX, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(tonemap);

static const AVFilterPad tonemap_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad tonemap_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_tonemap = {
    .name            = "tonemap",
    .description     = NULL_IF_CONFIG_SMALL("Conversion to/from different dynamic ranges."),
    .init            = init,
    .uninit          = uninit,
    .query_formats   = query_formats,
    .priv_size       = sizeof(TonemapContext),
    .priv_class      = &tonemap_class,
    .inputs          = tonemap_inputs,
    .outputs         = tonemap_outputs,
    .flags           = AVFILTER_FLAG_SLICE_THREADS,
};
