/*****************************************************************************

        Eedi3Sse.cpp

Optimizations for SSE/SSE2:

Process 8 lines at once.
First make a buffer of a capacity of 8*N lines, where N is the number of source
lines actually needed for processing a single output line (here: 4).
The PlanarFrame thing could be bypassed here when vcheck is 0.
Fill the buffer in the following way:
  For each source line offset o (-3, -1, +1, +3)
    For each column x (in 0..width-1, with a margin if necessary)
      For each line k in (0..7)
        Put the pixel src (x, y + o + k) in location (x * 8 + k, o)
        We could do the hpel interpolation here too.
Then do the exact C++ processing, but use vectors instead of scalars,
and multiply each horizontal coordinate by 8.
For AA purpose, we could interpolate columns with no additional transpose
cost (but what about the vcheck?)

With 8-bit input, the connection cost could be computed in 16-bit integer.
We could use multipliers like 32 or even 16 to get alpha and beta as integer
(they probably don't need a high accuracy).

Question: The C++ code uses double precision in the path cost section.
Is it really needed? The results are stored as float anyway...
We could go just with float and use single precision SEE registers.
Float overflow shouldn't be a concern here.
The visible problem is that the path cost loop acts like an integrator
of positive values, so the loss of precision could be an issue.
Test to conduct: check what is the maximum pcost value for a 8K picture
full of very contrasted hard edges and important background noise.

The inner loop in the path cost calculation could benefit from unrolling
(index clipped by u-1 and u+1).

The backtrack can be done only in C++ because of the indexing.

The "block" part of the algorithm would benefit from AVX2 (everything done
in one pass).

Other notes:

With the C++ version, 16-bit data interpolation could be done almost for free,
just templatize the function to use uint8_t/short as input and output
data type and saturate the result to the correct values.

TO DO:

- Understand and modify the vcheck code in order to remove the intermediate
  copies if possible.
- 16-bit input/output
- hpel interpolation
- Check the amount of horizontal margins really required. Why 12?
  It seems that the margin doesn't need to be greater than nrad (max 3).
- Optimize the core code of prepare/copy_result_lines*() with SSE2 code
  (transpose matrix)
- Treat the (expanded) mask differently, making it work like if masked areas
  were picture boundaries (precompute umax for every x). This should save
  more CPU.


Copyright (C) 2010 Kevin Stone - some part by Laurent de Soras, 2013

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*Tab=3***********************************************************************/



#if defined (_MSC_VER)
#pragma warning (1 : 4130 4223 4705 4706)
#pragma warning (4 : 4355 4786 4800)
#endif



/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

#include	"Eedi3Sse.h"

#include	<algorithm>

#include	<cassert>
#include	<cfloat>
#include	<cstring>



static __forceinline __m128i	difabs16(__m128i a, __m128i b)
{
  return (_mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a)));
}



/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/



/*
==============================================================================
Name: prepare_lines_8bits
Description:
  Reorders data before processing a set of lines.
  At most COL_H (8) lines will be processed per call.
Input parameters:
  - dst_pitch: In columns. One column = 8 pixels from 8 interpolated lines
    on the same abscissa.
  - src_ptr: Points on the top first *existing* reference line.
  - src_pitch: In bytes. Strides from one reference line to the next one.
  - bpp: Bytes per pixel (source). 1 for planar, or the horizontal step for
    interleaved formats.
  - width: Source width, in pixels
  - height: Number of existing reference lines
  - src_y: third reference line of the group of 4*8 to prepare.
    The interpolated line is located between the second and the third.
Output parameters:
  - dst_ptr: contains 4 reference lines (existing or mirrored) made of
    width + MARGIN_H * 2 columns. Points on the left margin.
Throws: Nothing
==============================================================================
*/

void	Eedi3Sse::prepare_lines_8bits(uint16_t *dst_ptr, int dst_pitch, const uint8_t *src_ptr, int src_pitch, int bpp, int width, int height, int src_y)
{
  assert(dst_ptr != 0);
  assert(dst_pitch > 0);
  assert(src_ptr != 0);
  assert(src_pitch > 0);
  assert(bpp > 0);
  assert(width > 0);
  assert(width * bpp <= src_pitch);
  assert(width + MARGIN_H * 2 <= dst_pitch);
  assert(height > 0);
  assert(src_y >= 0);
  assert(src_y <= height);

  for (int y = src_y - 2; y < src_y + 2; ++y)
  {
    uint16_t *     dst2_ptr = dst_ptr;

    for (int y2 = y; y2 < y + COL_H; ++y2)
    {
      int            real_y = (y2 < 0) ? -1 - y2 : y2;
      if (real_y >= height)
      {
        real_y = height * 2 - 1 - real_y;
      }
      real_y = std::max(real_y, 0);

      const uint8_t* line_ptr = src_ptr + real_y * src_pitch;

      for (int x = 0; x < MARGIN_H; ++x)
      {
        const int      src_x = std::min(MARGIN_H - 1 - x, width - 1);
        const uint16_t pix = line_ptr[src_x * bpp] << 8;
        dst2_ptr[x * COL_H] = pix;
      }
      for (int x = 0; x < width; ++x)
      {
        const uint16_t pix = line_ptr[x * bpp] << 8;
        dst2_ptr[(MARGIN_H + x) * COL_H] = pix;
      }
      for (int x = 0; x < MARGIN_H; ++x)
      {
        const int      src_x = std::max(width - 1 - x, 0);
        const uint16_t pix = line_ptr[src_x * bpp] << 8;
        dst2_ptr[(MARGIN_H + width + x) * COL_H] = pix;
      }

      ++dst2_ptr;
    }

    dst_ptr += dst_pitch * COL_H;
  }
}



void	Eedi3Sse::prepare_mask_8bits(uint8_t *dst_ptr, const uint8_t *src_ptr, int src_pitch, int bpp, int width, int height, int src_y)
{
  assert(dst_ptr != 0);
  assert(src_ptr != 0);
  assert(src_pitch > 0);
  assert(bpp > 0);
  assert(width > 0);
  assert(width * bpp <= src_pitch);
  assert(height > 0);
  assert(src_y >= 0);
  assert(src_y <= height);

  for (int y = src_y; y < src_y + COL_H; ++y)
  {
    int            real_y = y;
    if (real_y >= height)
    {
      real_y = height * 2 - 1 - real_y;
    }
    real_y = std::max(real_y, 0);

    const uint8_t* line_ptr = src_ptr + real_y * src_pitch;

    for (int x = 0; x < width; ++x)
    {
      dst_ptr[x * COL_H] = line_ptr[x * bpp];
    }

    ++dst_ptr;
  }
}



/*
==============================================================================
Name: copy_result_lines_8bits
Description:
  Unpack the result lines to the destination frame.
  At most COL_H (8) lines are processed per call.
  16-bit data is rounded to 8 bits for implementation simplicity.
Input parameters:
  - dst_pitch: In bytes. Strides from one line of the destination frame to
    the next one.
  - src_ptr: Pointer on the interpolation result data.
  - src_pitch: In columns. One column = 8 pixels from 8 interpolated lines
    on the same abscissa.
  - bpp: Bytes per pixel (destination). 1 for planar, or the horizontal step
    for interleaved formats.
  - width: Destination frame width, in pixels
  - height: Destination frame height, in pixels
  - dst_y: Position of the first line to unpack.
Output parameters:
  - dst_ptr: Pointer on the top left of the destination frame (no margin).
Throws: Nothing
==============================================================================
*/

void	Eedi3Sse::copy_result_lines_8bits(uint8_t *dst_ptr, int dst_pitch, const uint16_t *src_ptr, int src_pitch, int bpp, int width, int height, int dst_y)
{
  assert(dst_ptr != 0);
  assert(dst_pitch > 0);
  assert(src_ptr != 0);
  assert(src_pitch > 0);
  assert(bpp > 0);
  assert(width > 0);
  assert(width * bpp <= dst_pitch);
  assert(width <= src_pitch);
  assert(height > 0);
  assert(dst_y >= 0);
  assert(dst_y < height);

  const int      y_end = std::min(dst_y + COL_H, height);
  for (int y2 = dst_y; y2 < y_end; ++y2)
  {
    uint8_t *      line_ptr = dst_ptr + y2 * dst_pitch;

    for (int x = 0; x < width; ++x)
    {
      const uint16_t pix = src_ptr[x * COL_H];
      line_ptr[x * bpp] = std::min(std::max((pix + 0x80) >> 8, 0), 255);
    }

    ++src_ptr;
  }
}



void	Eedi3Sse::copy_result_dmap(int16_t *dst_ptr, int dst_pitch, const int16_t *src_ptr, int src_pitch, int width, int height, int dst_y)
{
  assert(dst_ptr != 0);
  assert(dst_pitch > 0);
  assert(src_ptr != 0);
  assert(src_pitch > 0);
  assert(width > 0);
  assert(width <= dst_pitch);
  assert(width <= src_pitch);
  assert(height > 0);
  assert(dst_y >= 0);
  assert(dst_y < height);

  const int      y_end = std::min(dst_y + COL_H, height);
  for (int y2 = dst_y; y2 < y_end; ++y2)
  {
    int16_t *      line_ptr = dst_ptr + y2 * dst_pitch;

    for (int x = 0; x < width; ++x)
    {
      line_ptr[x] = src_ptr[x * COL_H];
    }

    ++src_ptr;
  }
}



/*
==============================================================================
Name: interp_lines_full_pel
Description:
  Interpolates COL_H lines at once, full-pixel precision.
  Data must be packed with prepare_lines*() and the result unpacked with
  copy_result_lines*().
Input parameters:
  - src_ptr: Pointer on an array of 4 source lines with their margins
    (MARGIN_H pixels on both sides)
    Each line actually contains 8 packed lines, so each __m128i vector is
    made of 8 16-bit unsigned pixels from 8 different lines, on the same
    abscissa.
    It must point to the first left pixel of the left margin.
    The interpolated line will be located between the second and the third
    source lines.
  - msk_ptr: A pointer on mask data, or 0 if all pixels must be processed.
    Mask data contains 8 lines of boolean bytes (0 or != 0), packed by 8
    pixels to follow the output format.
    The mask hasn't any margin.
  - width: Number of pixels (or __m128 units) to process.
  - pitch: Pitch of the source in pixels (or __m128 units). Should obviously
    take the left and right margins into account.
  - alpha: See user documentation.
  - beta: See user documentation.
  - gamma: See user documentation.
  - nrad: See user documentation.
  - mdis: See user documentation.
  - ucubic: See user documentation.
  - cost3: See user documentation.
Output parameters:
  - dst_ptr: Pointer on a buffer receiving the interpolated line.
    8 packed unsigned 16-bit pixels per vector.
    Only unmasked pixels are valid.
  - dmap_ptr: A buffer of signed 16-bit int corresponding to the interpolated
    pixels, in packed form again.
    The values give the horizontal slope of the interpolation direction for
    each pixel and are later used in the vcheck pass.
    Only unmasked pixels are valid.
Input/output parameters:
  - tmp_ptr: A temporary buffer. Its size in bytes is at least:
    ((4 * (mdis * 2 + 1) + 1) * VECTSIZE * 4 + 1) * width
Throws: Nothing
==============================================================================
*/

void	Eedi3Sse::interp_lines_full_pel(const __m128i *src_ptr, __m128i *dst_ptr, const uint8_t *msk_ptr, uint8_t *tmp_ptr, __m128i *dmap_ptr, int width, int pitch, float alpha, float beta, float gamma, int nrad, int mdis, bool ucubic, bool cost3)
{
  assert(src_ptr != 0);
  assert(dst_ptr != 0);
  assert(tmp_ptr != 0);
  assert(dmap_ptr != 0);
  assert(width > 0);
  assert(pitch > 0);
  assert(alpha >= 0);
  assert(beta >= 0);
  assert(alpha + beta <= 1.0f);
  assert(gamma >= 0);
  assert(nrad >= 0);
  assert(nrad <= 3);
  assert(mdis > 0);

  // First, shifts everything so we point on actual data.
  src_ptr += MARGIN_H;

  const __m128i* src3p_ptr = src_ptr;
  const __m128i* src1p_ptr = src_ptr + 1 * pitch;
  const __m128i* src1n_ptr = src_ptr + 2 * pitch;
  const __m128i* src3n_ptr = src_ptr + 3 * pitch;

  /*** Debugging code *********************************************************/
#if 0 // EDI bypass (simple cubic interpolation)
  const __m128i  zero = _mm_setzero_si128();
  const __m128i	nine16 = _mm_set1_epi16(9);
  const __m128i	sign16 = _mm_set1_epi16(-0x8000);
  const __m128i	cubic_cst = _mm_set1_epi32(-0x8000 * 8 + 4); // Rounding and sign change
  for (int x = 0; x < width; ++x)
  {
    __m128i s1p = _mm_load_si128(src1p_ptr + x);
    __m128i s1n = _mm_load_si128(src1n_ptr + x);
    __m128i s3p = _mm_load_si128(src3p_ptr + x);
    __m128i s3n = _mm_load_si128(src3n_ptr + x);
    _mm_store_si128(dst_ptr + x, interp_cubic8(
      s1p, s1n, s3p, s3n, nine16, sign16, cubic_cst, zero
    ));
    _mm_store_si128(dmap_ptr + x, zero);
  }
#else
/****************************************************************************/

  const int      tpitch = mdis * 2 + 1;
  int            tmpofs = 0;

#define Eedi3Sse_DECL( T, N, S) \
	T  *          N = reinterpret_cast <T *> (tmp_ptr + tmpofs); \
	tmpofs += S * sizeof (T);

  // ccosts is grouped in 2 separate chunks of 4 packed lines,
  // pcosts, pbackt and fpath are chunks of 4 packed lines,
  // bmask contains a single boolean for each 8-line column
  Eedi3Sse_DECL(float, ccosts, 2 * width * tpitch * VECTSIZE); // Array of mdis*2+1 costs for each pixel of the line
  Eedi3Sse_DECL(float, pcosts, width * tpitch * VECTSIZE);
  Eedi3Sse_DECL(int32_t, pbackt, width * tpitch * VECTSIZE);
  Eedi3Sse_DECL(int32_t, fpath, width          * VECTSIZE);
  Eedi3Sse_DECL(bool, bmask, width);

#undef Eedi3Sse_DECL

  if (msk_ptr != 0)
  {
    memset(ccosts, 0, 2 * tpitch * width * VECTSIZE * sizeof(float));
    expand_mask(bmask, msk_ptr, width, mdis);
  }

  const __m128i  zero = _mm_setzero_si128();
  const __m128   alpha_4 = _mm_set1_ps((cost3) ? alpha / 3.f : alpha);
  const __m128   ab_4 = _mm_set1_ps(1.0f - alpha - beta);

  const int      tpitch_v = tpitch * VECTSIZE;
  const int      ofs_p4 = width * tpitch_v;

  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  // Calculate all connection costs

  // beta is calibrated for 8-bit content. We have to scale it because it is
  // not multiplied by a pixel value in the formula, contrary to the global
  // result which depends on the data scale.
  const float    beta16 = beta * 256;

  if (!cost3)
  {
    for (int x = 0; x < width; ++x)
    {
      if (msk_ptr == 0 || bmask[x] != 0)
      {
        const int      umax = std::min(std::min(x, width - 1 - x), mdis);
        for (int u = -umax; u <= umax; ++u)
        {
          __m128i        s_0;
          __m128i			s_1;
          sum_nrad(
            s_0, s_1, zero, nrad, x - u, x + u,
            src3p_ptr, src1p_ptr, src1n_ptr, src3n_ptr
          );

          // should use cubic if ucubic=true
          const __m128i  x1pr = _mm_load_si128(src1p_ptr + x + u);
          const __m128i  x1nl = _mm_load_si128(src1n_ptr + x - u);
          const __m128i  ip = _mm_avg_epu16(x1pr, x1nl);

          const __m128i  x1p = _mm_load_si128(src1p_ptr + x);
          const __m128i  x1n = _mm_load_si128(src1n_ptr + x);
          const __m128i  vdp = difabs16(x1p, ip);
          const __m128i  vdn = difabs16(x1n, ip);
          const __m128i  v0 = _mm_add_epi32(
            _mm_unpacklo_epi16(vdp, zero),
            _mm_unpacklo_epi16(vdn, zero)
          );
          const __m128i  v1 = _mm_add_epi32(
            _mm_unpackhi_epi16(vdp, zero),
            _mm_unpackhi_epi16(vdn, zero)
          );

          const __m128   cc_a0 = _mm_mul_ps(_mm_cvtepi32_ps(s_0), alpha_4);
          const __m128   cc_a1 = _mm_mul_ps(_mm_cvtepi32_ps(s_1), alpha_4);

          const __m128   cc_b = _mm_set1_ps(beta16 * std::abs(u));

          const __m128   cc_c0 = _mm_mul_ps(_mm_cvtepi32_ps(v0), ab_4);
          const __m128   cc_c1 = _mm_mul_ps(_mm_cvtepi32_ps(v1), ab_4);

          const __m128   cc0 = _mm_add_ps(_mm_add_ps(cc_a0, cc_b), cc_c0);
          const __m128   cc1 = _mm_add_ps(_mm_add_ps(cc_a1, cc_b), cc_c1);
          const int      dpos = (x * tpitch + mdis + u) * VECTSIZE;
          _mm_store_ps(ccosts + dpos, cc0);
          _mm_store_ps(ccosts + dpos + ofs_p4, cc1);
        }
      }
    }
  }
  else	// cost3
  {
    for (int x = 0; x < width; ++x)
    {
      if (msk_ptr == 0 || bmask[x] != 0)
      {
        const int      umax = std::min(std::min(x, width - 1 - x), mdis);
        for (int u = -umax; u <= umax; ++u)
        {
          const bool     s1_flag = ((u >= 0 && x >= u * 2)
            || (u <= 0 && x < width + u * 2));
          const bool     s2_flag = ((u <= 0 && x >= u * -2)
            || (u >= 0 && x < width + u * 2));

          __m128i        s0_0;
          __m128i        s0_1;
          sum_nrad(
            s0_0, s0_1, zero, nrad, x - u, x + u,
            src3p_ptr, src1p_ptr, src1n_ptr, src3n_ptr
          );

          __m128i        s1_0;
          __m128i        s1_1;
          if (s1_flag)
          {
            sum_nrad(
              s1_0, s1_1, zero, nrad, x - 2 * u, x,
              src3p_ptr, src1p_ptr, src1n_ptr, src3n_ptr
            );
          }

          __m128i        s2_0;
          __m128i        s2_1;
          if (s2_flag)
          {
            sum_nrad(
              s2_0, s2_1, zero, nrad, x, x + 2 * u,
              src3p_ptr, src1p_ptr, src1n_ptr, src3n_ptr
            );
          }

          s1_0 = (s1_flag) ? s1_0 : ((s2_flag) ? s2_0 : s0_0);
          s1_1 = (s1_flag) ? s1_1 : ((s2_flag) ? s2_1 : s0_1);
          s2_0 = (s2_flag) ? s2_0 : ((s1_flag) ? s1_0 : s0_0);
          s2_1 = (s2_flag) ? s2_1 : ((s1_flag) ? s1_1 : s0_1);

          const __m128i  s_0 = _mm_add_epi32(_mm_add_epi32(s0_0, s1_0), s2_0);
          const __m128i  s_1 = _mm_add_epi32(_mm_add_epi32(s0_1, s1_1), s2_1);

          // should use cubic if ucubic=true
          const __m128i  x1pr = _mm_load_si128(src1p_ptr + x + u);
          const __m128i  x1nl = _mm_load_si128(src1n_ptr + x - u);
          const __m128i  ip = _mm_avg_epu16(x1pr, x1nl);

          const __m128i  x1p = _mm_load_si128(src1p_ptr + x);
          const __m128i  x1n = _mm_load_si128(src1n_ptr + x);
          const __m128i  vdp = difabs16(x1p, ip);
          const __m128i  vdn = difabs16(x1n, ip);
          const __m128i  v0 = _mm_add_epi32(
            _mm_unpacklo_epi16(vdp, zero),
            _mm_unpacklo_epi16(vdn, zero)
          );
          const __m128i  v1 = _mm_add_epi32(
            _mm_unpackhi_epi16(vdp, zero),
            _mm_unpackhi_epi16(vdn, zero)
          );

          const __m128   cc_a0 = _mm_mul_ps(_mm_cvtepi32_ps(s_0), alpha_4);
          const __m128   cc_a1 = _mm_mul_ps(_mm_cvtepi32_ps(s_1), alpha_4);

          const __m128   cc_b = _mm_set1_ps(beta16 * std::abs(u));

          const __m128   cc_c0 = _mm_mul_ps(_mm_cvtepi32_ps(v0), ab_4);
          const __m128   cc_c1 = _mm_mul_ps(_mm_cvtepi32_ps(v1), ab_4);

          const __m128   cc0 = _mm_add_ps(_mm_add_ps(cc_a0, cc_b), cc_c0);
          const __m128   cc1 = _mm_add_ps(_mm_add_ps(cc_a1, cc_b), cc_c1);
          const int      dpos = (x * tpitch + mdis + u) * VECTSIZE;
          _mm_store_ps(ccosts + dpos, cc0);
          _mm_store_ps(ccosts + dpos + ofs_p4, cc1);
        }
      }
    }
  }

  const __m128   fltmax = _mm_set1_ps(FLT_MAX);
  const __m128   fltmax9 = _mm_set1_ps(FLT_MAX * 0.9f);
  const __m128i	nine16 = _mm_set1_epi16(9);
  const __m128i	sign16 = _mm_set1_epi16(-0x8000);
  const __m128i	cubic_cst = _mm_set1_epi32(-0x8000 * 8 + 4); // Rounding and sign change

  // Same reason as beta16
  const float    gamma16 = gamma * 256;

  // The following operations are done in 2 passes (the "blocks"), because
  // we can process only VECTSIZE pixels at once (FP32 data).
  // Note: ccosts pointer is shifted at the end of the block
  for (int block = 0; block < 2; ++block)
  {
    // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    // Calculate path costs

    for (int k = 0; k < VECTSIZE; ++k)
    {
      const int      p = mdis * VECTSIZE + k;
      pcosts[p] = ccosts[p];
    }

    for (int x = 1; x < width; ++x)
    {
      float *        tT = ccosts + x * tpitch_v;
      float *        ppT = pcosts + (x - 1) * tpitch_v;
      float *        pT = pcosts + x * tpitch_v;
      __m128i *      piT = reinterpret_cast <__m128i *> (pbackt + (x - 1) * tpitch_v);

      if (msk_ptr != 0 && bmask[x] == 0)
      {
        if (x == 1)
        {
          const int      umax = std::min(std::min(x, width - 1 - x), mdis);
          const int      p = (mdis - umax) * VECTSIZE;
          memcpy(pT + p, tT + p, (umax * 2 + 1) * VECTSIZE * sizeof(*pT));
          memset(piT, 0, tpitch * sizeof(*piT));
        }
        else
        {
          memcpy(pT, ppT, tpitch_v * sizeof(*pT));
          memcpy(piT, piT - tpitch, tpitch * sizeof(*piT));
          const int      pumax = std::min(x - 1, width - x);
          if (pumax < mdis)
          {
            const __m128i  a = _mm_set1_epi32(1 - pumax);
            const __m128i  b = _mm_set1_epi32(pumax - 1);
            _mm_store_si128(piT + mdis - pumax, a);
            _mm_store_si128(piT + mdis + pumax, b);
          }
        }
      }

      else
      {
        const int      umax = std::min(std::min(x, width - 1 - x), mdis);
        for (int u = -umax; u <= umax; ++u)
        {
          __m128i        idx = _mm_setzero_si128(); // 32-bit signed int
          __m128         bval = fltmax;
          const int      umax2 = std::min(std::min(x - 1, width - x), mdis);
          const int      vmax = std::min(umax2, u + 1);
          for (int v = std::max(-umax2, u - 1); v <= vmax; ++v)
          {
            __m128         y = _mm_load_ps(ppT + (mdis + v) * VECTSIZE);
            const __m128   a = _mm_set1_ps(gamma16 * std::abs(u - v));
            y = _mm_add_ps(y, a);
            const __m128   ccost = _mm_min_ps(y, fltmax9);
            const __m128i  v4 = _mm_set1_epi32(v);
            const __m128i  tst =  // if (ccost < bval)
              _mm_castps_si128(_mm_cmplt_ps(ccost, bval));
            idx = select(tst, v4, idx);
            bval = _mm_min_ps(ccost, bval);
          }
          const int      mu = (mdis + u) * VECTSIZE;
          __m128         y = _mm_add_ps(bval, _mm_load_ps(tT + mu));
          y = _mm_min_ps(y, fltmax9);
          _mm_store_ps(pT + mu, y);
          _mm_store_si128(piT + mdis + u, idx);
        }
      }
    }

    // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    // Backtrack

    _mm_store_si128(reinterpret_cast <__m128i *> (fpath) + width - 1, zero);
    for (int x = width - 2; x >= 0; --x)
    {
      const int      idx_n = (x + 1)             * VECTSIZE;
      const int      idx_c = x * VECTSIZE;
      const int      idx_p = (x * tpitch + mdis) * VECTSIZE;
      for (int k = 0; k < VECTSIZE; ++k)
      {
        const int      n = fpath[idx_n + k];
        fpath[idx_c + k] = pbackt[idx_p + n * VECTSIZE + k];
      }
    }

    // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    // Interpolate

    const int      bv = block * VECTSIZE;
    for (int x = 0; x < width; ++x)
    {
      if (msk_ptr != 0 && bmask[x] == 0)
      {
        // Does both blocks at once.
        if (block == 0)
        {
          _mm_store_si128(dmap_ptr + x, zero);

          __m128i        res;
          if (ucubic)
          {
            res = interp_cubic8(
              _mm_load_si128(src1p_ptr + x),
              _mm_load_si128(src1n_ptr + x),
              _mm_load_si128(src3p_ptr + x),
              _mm_load_si128(src3n_ptr + x),
              nine16, sign16, cubic_cst, zero
            );
          }
          else
          {
            res = _mm_avg_epu16(
              _mm_load_si128(src1p_ptr + x),
              _mm_load_si128(src1n_ptr + x)
            );
          }
          _mm_store_si128(dst_ptr + x, res);
        }
      }

      else
      {
        uint16_t * const       dst16_ptr =
          reinterpret_cast <uint16_t *> (dst_ptr + x) + bv;

        assert(sizeof(*dmap_ptr) == sizeof(int16_t) * VECTSIZE * 2);
        assert(sizeof(*fpath) == sizeof(int32_t));
        __m128i        dir4 =
          _mm_load_si128(reinterpret_cast <const __m128i *> (fpath) + x);
        dir4 = _mm_packs_epi32(dir4, zero);   // Contains 16-bit data
        _mm_storel_epi64(reinterpret_cast <__m128i *> (
          reinterpret_cast <int64_t *> (dmap_ptr + x) + block
          ), dir4);

        /*** To do: use interp_cubic4() and check if it's faster ***/

        for (int k = 0; k < VECTSIZE; ++k)
        {
          // Cast to int16_t because _mm_extract_epi16 extends with 0s
          // and we need the sign.
          const int      dir = int16_t(_mm_extract_epi16(dir4, 0));

          const uint16_t * const src1p16_ptr =
            reinterpret_cast <const uint16_t *> (src1p_ptr + x + dir) + bv;
          const uint16_t * const src1n16_ptr =
            reinterpret_cast <const uint16_t *> (src1n_ptr + x - dir) + bv;
          const int      sum_1 = src1p16_ptr[k] + src1n16_ptr[k];

          const int      ad = std::abs(dir);
          if (ucubic && x >= ad * 3 && x <= width - 1 - ad * 3)
          {
            const uint16_t * const src3p16_ptr =
              reinterpret_cast <const uint16_t *> (src3p_ptr + x + dir * 3) + bv;
            const uint16_t * const src3n16_ptr =
              reinterpret_cast <const uint16_t *> (src3n_ptr + x - dir * 3) + bv;
            const int      sum_3 = src3p16_ptr[k] + src3n16_ptr[k];

            const int      interp = (9 * sum_1 - sum_3 + 8) >> 4;
            dst16_ptr[k] =
              uint16_t(std::min(std::max(interp, 0), 65535));
          }
          else
          {
            dst16_ptr[k] = uint16_t((sum_1 + 1) >> 1);
          }

          dir4 = _mm_srli_si128(dir4, 2);
        }
      }
    }  // for x

    ccosts += ofs_p4;

  }  // for block
/****************************************************************************/
#endif // EDI bypass
/****************************************************************************/
}



/*\\\ PROTECTED \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/



/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/



void	Eedi3Sse::expand_mask(bool dst_ptr[], const uint8_t msk_ptr[], int width, int mdis)
{
  assert(dst_ptr != 0);
  assert(msk_ptr != 0);
  assert(width > 0);
  assert(mdis > 0);

  const int64_t *   msk8_ptr = reinterpret_cast <const int64_t *> (msk_ptr);

  const int	minmdis = (width < mdis) ? width : mdis;

  int			last = -666999;

  for (int x = 0; x < minmdis; ++x)
  {
    if (msk8_ptr[x] != 0)
    {
      last = x + mdis;
    }
  }

  for (int x = 0; x < width - minmdis; ++x)
  {
    if (msk8_ptr[x + mdis] != 0)
    {
      last = x + mdis * 2;
    }
    dst_ptr[x] = (x <= last);
  }

  for (int x = width - minmdis; x < width; ++x)
  {
    dst_ptr[x] = (x <= last);
  }
}



void	Eedi3Sse::sum_nrad(__m128i &s_0, __m128i &s_1, const __m128i &zero, int nrad, int xl, int xr, const __m128i *src3p, const __m128i *src1p, const __m128i *src1n, const __m128i *src3n)
{
  assert(nrad >= 0);
  assert(src3p != 0);
  assert(src1p != 0);
  assert(src1n != 0);
  assert(src3n != 0);

  s_0 = zero;
  s_1 = zero;
  for (int k = -nrad; k <= nrad; ++k)
  {
    const int      xrk = xr + k;
    const int      xlk = xl + k;
    const __m128i  x3pr = _mm_load_si128(src3p + xrk);
    const __m128i  x1pl = _mm_load_si128(src1p + xlk);
    const __m128i  x1pr = _mm_load_si128(src1p + xrk);
    const __m128i  x1nl = _mm_load_si128(src1n + xlk);
    const __m128i  x1nr = _mm_load_si128(src1n + xrk);
    const __m128i  x3nl = _mm_load_si128(src3n + xlk);

    const __m128i  d3p1p = difabs16(x3pr, x1pl);
    const __m128i  d1p1n = difabs16(x1pr, x1nl);
    const __m128i  d1n3n = difabs16(x1nr, x3nl);

    s_0 = _mm_add_epi32(s_0, _mm_unpacklo_epi16(d3p1p, zero));
    s_1 = _mm_add_epi32(s_1, _mm_unpackhi_epi16(d3p1p, zero));
    s_0 = _mm_add_epi32(s_0, _mm_unpacklo_epi16(d1p1n, zero));
    s_1 = _mm_add_epi32(s_1, _mm_unpackhi_epi16(d1p1n, zero));
    s_0 = _mm_add_epi32(s_0, _mm_unpacklo_epi16(d1n3n, zero));
    s_1 = _mm_add_epi32(s_1, _mm_unpackhi_epi16(d1n3n, zero));
  }
}



__m128i	Eedi3Sse::interp_cubic8(const __m128i &src1p, const __m128i &src1n, const __m128i &src3p, const __m128i &src3n, const __m128i &nine16, const __m128i &sign16, const __m128i &cubic_cst, const __m128i &zero)
{
  assert(&src1p != 0);
  assert(&src1n != 0);
  assert(&src3p != 0);
  assert(&src3n != 0);
  assert(&nine16 != 0);
  assert(&sign16 != 0);
  assert(&cubic_cst != 0);
  assert(&zero != 0);

  const __m128i  avg1 = _mm_avg_epu16(src1p, src1n);
  const __m128i  avg3 = _mm_avg_epu16(src3p, src3n);
  const __m128i  a3_0 = _mm_unpacklo_epi16(avg3, zero);
  const __m128i  a3_1 = _mm_unpackhi_epi16(avg3, zero);
  const __m128i  hi = _mm_mulhi_epu16(avg1, nine16);
  const __m128i  lo = _mm_mullo_epi16(avg1, nine16);
  __m128i        s0 = _mm_unpacklo_epi16(lo, hi);
  __m128i        s1 = _mm_unpackhi_epi16(lo, hi);
  s0 = _mm_sub_epi32(s0, a3_0);
  s1 = _mm_sub_epi32(s1, a3_1);
  s0 = _mm_add_epi32(s0, cubic_cst);
  s1 = _mm_add_epi32(s1, cubic_cst);
  s0 = _mm_srai_epi32(s0, 3);
  s1 = _mm_srai_epi32(s1, 3);
  __m128i        res = _mm_packs_epi32(s0, s1);
  res = _mm_xor_si128(res, sign16);

  return (res);
}



// src13p and src13n are made of:
// - 4 int16 from 1p/1n in the lowest 64 bits, and
// - 4 int16 from 3p/3n in the highest 64 bits.
// Result is packed in the lowest 64 bits, highest 64 bits are garbage.
__m128i	Eedi3Sse::interp_cubic4(const __m128i &src13p, const __m128i &src13n, const __m128i &nine16, const __m128i &sign16, const __m128i &cubic_cst, const __m128i &zero)
{
  assert(&src13p != 0);
  assert(&src13n != 0);
  assert(&nine16 != 0);
  assert(&sign16 != 0);
  assert(&cubic_cst != 0);
  assert(&zero != 0);

  const __m128i  avg = _mm_avg_epu16(src13p, src13n);
  const __m128i  a3 = _mm_unpackhi_epi16(avg, zero);
  const __m128i  hi = _mm_mulhi_epu16(avg, nine16);
  const __m128i  lo = _mm_mullo_epi16(avg, nine16);
  __m128i        s = _mm_unpacklo_epi16(lo, hi);
  s = _mm_sub_epi32(s, a3);
  s = _mm_add_epi32(s, cubic_cst);
  s = _mm_srai_epi32(s, 3);
  __m128i        res = _mm_packs_epi32(s, zero);
  res = _mm_xor_si128(res, sign16);

  return (res);
}



__m128i	Eedi3Sse::select(const __m128i &cond, const __m128i &v_t, const __m128i &v_f)
{
  const __m128i  cond_1 = _mm_and_si128(cond, v_t);
  const __m128i  cond_0 = _mm_andnot_si128(cond, v_f);
  const __m128i  res = _mm_or_si128(cond_0, cond_1);

  return (res);
}



/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
