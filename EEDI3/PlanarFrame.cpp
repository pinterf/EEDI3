/*
**   My PlanarFrame class... fast mmx/sse2 YUY2 packed to planar and planar
**   to packed conversions, and always gives 16 bit alignment for all
**   planes.  Supports YV12/YUY2/RGB24 frames from avisynth, can do any planar
**   format internally.
**
**   Copyright (C) 2005-2010 Kevin Stone
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "PlanarFrame.h"
#include <stdint.h>
#include <emmintrin.h>

int modnpf(const int m, const int n)
{
  if ((m%n) == 0)
    return m;
  return m + n - (m%n);
}

PlanarFrame::PlanarFrame(int cpuFlags)
{
  ypitch = uvpitch = 0;
  ywidth = uvwidth = 0;
  yheight = uvheight = 0;
  y = u = v = NULL;
  useSIMD = true;
  cpu = cpuFlags;
}

PlanarFrame::PlanarFrame(VideoInfo &viInfo, int cpuFlags)
{
  ypitch = uvpitch = 0;
  ywidth = uvwidth = 0;
  yheight = uvheight = 0;
  y = u = v = NULL;
  useSIMD = true;
  cpu = cpuFlags;
  allocSpace(viInfo);
}

PlanarFrame::~PlanarFrame()
{
  if (y != NULL) { _aligned_free(y); y = NULL; }
  if (u != NULL) { _aligned_free(u); u = NULL; }
  if (v != NULL) { _aligned_free(v); v = NULL; }
}

bool PlanarFrame::allocSpace(VideoInfo &viInfo)
{
  if (y != NULL) { _aligned_free(y); y = NULL; }
  if (u != NULL) { _aligned_free(u); u = NULL; }
  if (v != NULL) { _aligned_free(v); v = NULL; }
  ypitch = uvpitch = 0;
  ywidth = uvwidth = 0;
  yheight = uvheight = 0;

  int height = viInfo.height;
  int width = viInfo.width;
  if ((height == 0) || (width == 0)) return false;
  if (viInfo.IsYV12())
  {
    ypitch = modnpf(width + MIN_PAD, MIN_ALIGNMENT);
    ywidth = width;
    yheight = height;
    width >>= 1;
    height >>= 1;
    uvpitch = modnpf(width + MIN_PAD, MIN_ALIGNMENT);
    uvwidth = width;
    uvheight = height;
  }
  else
  {
    if (viInfo.IsYUY2() || viInfo.IsYV16())
    {
      ypitch = modnpf(width + MIN_PAD, MIN_ALIGNMENT);
      ywidth = width;
      yheight = height;
      width >>= 1;
      uvpitch = modnpf(width + MIN_PAD, MIN_ALIGNMENT);
      uvwidth = width;
      uvheight = height;
    }
    else
    {
      if (viInfo.IsRGB24() || viInfo.IsYV24())
      {
        ypitch = modnpf(width + MIN_PAD, MIN_ALIGNMENT);
        ywidth = width;
        yheight = height;
        uvpitch = ypitch;
        uvwidth = ywidth;
        uvheight = yheight;
      }
      else
      {
        if (viInfo.IsY8())
        {
          ypitch = modnpf(width + MIN_PAD, MIN_ALIGNMENT);
          ywidth = width;
          yheight = height;
        }
      }
    }
  }
  y = (unsigned char*)_aligned_malloc(ypitch*yheight, MIN_ALIGNMENT);
  if (y == NULL) return false;
  if ((uvpitch != 0) && (uvheight != 0))
  {
    u = (unsigned char*)_aligned_malloc(uvpitch*uvheight, MIN_ALIGNMENT);
    if (u == NULL) return false;
    v = (unsigned char*)_aligned_malloc(uvpitch*uvheight, MIN_ALIGNMENT);
    if (v == NULL) return false;
  }
  return true;
}

bool PlanarFrame::allocSpace(int specs[4])
{
  if (y != NULL) { _aligned_free(y); y = NULL; }
  if (u != NULL) { _aligned_free(u); u = NULL; }
  if (v != NULL) { _aligned_free(v); v = NULL; }
  ypitch = uvpitch = 0;
  ywidth = uvwidth = 0;
  yheight = uvheight = 0;

  int height = specs[0];
  int width = specs[2];
  if ((height == 0) || (width == 0)) return false;
  ypitch = modnpf(width + MIN_PAD, MIN_ALIGNMENT);
  ywidth = width;
  yheight = height;
  height = specs[1];
  width = specs[3];
  if ((width != 0) && (height != 0))
  {
    uvpitch = modnpf(width + MIN_PAD, MIN_ALIGNMENT);
    uvwidth = width;
    uvheight = height;
  }
  y = (unsigned char*)_aligned_malloc(ypitch*yheight, MIN_ALIGNMENT);
  if (y == NULL) return false;
  if ((uvpitch != 0) && (uvheight != 0))
  {
    u = (unsigned char*)_aligned_malloc(uvpitch*uvheight, MIN_ALIGNMENT);
    if (u == NULL) return false;
    v = (unsigned char*)_aligned_malloc(uvpitch*uvheight, MIN_ALIGNMENT);
    if (v == NULL) return false;
  }
  return true;
}

void PlanarFrame::createPlanar(int yheight, int uvheight, int ywidth, int uvwidth)
{
  int specs[4] = { yheight,uvheight,ywidth,uvwidth };
  allocSpace(specs);
}

void PlanarFrame::createPlanar(int height, int width, uint8_t chroma_format)
{
  int specs[4];
  switch (chroma_format)
  {
  case 0:
  case 1:
    specs[0] = height; specs[1] = height >> 1;
    specs[2] = width; specs[3] = width >> 1;
    break;
  case 2:
    specs[0] = height; specs[1] = height;
    specs[2] = width; specs[3] = width >> 1;
    break;
  default:
    specs[0] = height; specs[1] = height;
    specs[2] = width; specs[3] = width;
    break;
  }
  allocSpace(specs);
}

void PlanarFrame::createFromProfile(VideoInfo &viInfo)
{
  allocSpace(viInfo);
}

void PlanarFrame::createFromFrame(PVideoFrame &frame, VideoInfo &viInfo)
{
  allocSpace(viInfo);
  copyInternalFrom(frame, viInfo);
}

void PlanarFrame::createFromPlanar(PlanarFrame &frame)
{
  int specs[4] = { frame.yheight,frame.uvheight,frame.ywidth,frame.uvwidth };
  allocSpace(specs);
  copyInternalFrom(frame);
}

void PlanarFrame::copyFrom(PVideoFrame &frame, VideoInfo &viInfo)
{
  copyInternalFrom(frame, viInfo);
}

void PlanarFrame::copyFrom(PlanarFrame &frame)
{
  copyInternalFrom(frame);
}

void PlanarFrame::copyTo(PVideoFrame &frame, VideoInfo &viInfo)
{
  copyInternalTo(frame, viInfo);
}

void PlanarFrame::copyTo(PlanarFrame &frame)
{
  copyInternalTo(frame);
}

void PlanarFrame::copyPlaneTo(PlanarFrame &frame, uint8_t plane)
{
  copyInternalPlaneTo(frame, plane);
}

uint8_t* PlanarFrame::GetPtr(uint8_t plane)
{
  switch (plane)
  {
  case 0: return y; break;
  case 1: return u; break;
  default: return v; break;
  }
}

int PlanarFrame::GetWidth(uint8_t plane)
{
  switch (plane)
  {
  case 0: return ywidth; break;
  default: return uvwidth; break;
  }
}

int PlanarFrame::GetHeight(uint8_t plane)
{
  switch (plane)
  {
  case 0: return yheight; break;
  default: return uvheight; break;
  }
}

int PlanarFrame::GetPitch(uint8_t plane)
{
  switch (plane)
  {
  case 0: return ypitch; break;
  default: return uvpitch; break;
  }
}

void PlanarFrame::freePlanar()
{
  if (y != NULL) { _aligned_free(y); y = NULL; }
  if (u != NULL) { _aligned_free(u); u = NULL; }
  if (v != NULL) { _aligned_free(v); v = NULL; }
  ypitch = uvpitch = 0;
  ywidth = uvwidth = 0;
  yheight = uvheight = 0;
  cpu = 0;
}

void PlanarFrame::copyInternalFrom(PVideoFrame &frame, VideoInfo &viInfo)
{
  if ((y == NULL) || (!viInfo.IsY8() && ((u == NULL) || (v == NULL)))) return;

  if (viInfo.IsYV12() || viInfo.IsYV16() || viInfo.IsYV24())
  {
    BitBlt(y, ypitch, frame->GetReadPtr(PLANAR_Y), frame->GetPitch(PLANAR_Y),
      frame->GetRowSize(PLANAR_Y), frame->GetHeight(PLANAR_Y));
    BitBlt(u, uvpitch, frame->GetReadPtr(PLANAR_U), frame->GetPitch(PLANAR_U),
      frame->GetRowSize(PLANAR_U), frame->GetHeight(PLANAR_U));
    BitBlt(v, uvpitch, frame->GetReadPtr(PLANAR_V), frame->GetPitch(PLANAR_V),
      frame->GetRowSize(PLANAR_V), frame->GetHeight(PLANAR_V));
  }
  else if (viInfo.IsY8())
  {
    BitBlt(y, ypitch, frame->GetReadPtr(PLANAR_Y), frame->GetPitch(PLANAR_Y),
      frame->GetRowSize(PLANAR_Y), frame->GetHeight(PLANAR_Y));
  }
  else if (viInfo.IsYUY2())
  {
    convYUY2to422(frame->GetReadPtr(), y, u, v, frame->GetPitch(), ypitch, uvpitch,
      viInfo.width, viInfo.height);
  }
  else
  {
    if (viInfo.IsRGB24())
    {
      convRGB24to444(frame->GetReadPtr(), y, u, v, frame->GetPitch(), ypitch, uvpitch,
        viInfo.width, viInfo.height);
    }
  }
}

void PlanarFrame::copyInternalFrom(PlanarFrame &frame)
{
  if ((y == NULL) || ((uvpitch != 0) && ((u == NULL) || (v == NULL)))) return;

  BitBlt(y, ypitch, frame.y, frame.ypitch, frame.ywidth, frame.yheight);
  if (uvpitch != 0)
  {
    BitBlt(u, uvpitch, frame.u, frame.uvpitch, frame.uvwidth, frame.uvheight);
    BitBlt(v, uvpitch, frame.v, frame.uvpitch, frame.uvwidth, frame.uvheight);
  }
}

void PlanarFrame::copyInternalTo(PVideoFrame &frame, VideoInfo &viInfo)
{
  if ((y == NULL) || (!viInfo.IsY8() && ((u == NULL) || (v == NULL)))) return;

  if (viInfo.IsYV12() || viInfo.IsYV16() || viInfo.IsYV24())
  {
    BitBlt(frame->GetWritePtr(PLANAR_Y), frame->GetPitch(PLANAR_Y), y, ypitch, ywidth, yheight);
    BitBlt(frame->GetWritePtr(PLANAR_U), frame->GetPitch(PLANAR_U), u, uvpitch, uvwidth, uvheight);
    BitBlt(frame->GetWritePtr(PLANAR_V), frame->GetPitch(PLANAR_V), v, uvpitch, uvwidth, uvheight);
  }
  else if (viInfo.IsY8())
  {
    BitBlt(frame->GetWritePtr(PLANAR_Y), frame->GetPitch(PLANAR_Y), y, ypitch, ywidth, yheight);
  }
  else if (viInfo.IsYUY2())
  {
    conv422toYUY2(y, u, v, frame->GetWritePtr(), ypitch, uvpitch, frame->GetPitch(), ywidth, yheight);
  }
  else if (viInfo.IsRGB24())
  {
    conv444toRGB24(y, u, v, frame->GetWritePtr(), ypitch, uvpitch, frame->GetPitch(), ywidth, yheight);
  }
}

void PlanarFrame::copyInternalTo(PlanarFrame &frame)
{
  if ((y == NULL) || ((uvpitch != 0) && ((u == NULL) || (v == NULL)))) return;

  BitBlt(frame.y, frame.ypitch, y, ypitch, ywidth, yheight);
  if (uvpitch != 0)
  {
    BitBlt(frame.u, frame.uvpitch, u, uvpitch, uvwidth, uvheight);
    BitBlt(frame.v, frame.uvpitch, v, uvpitch, uvwidth, uvheight);
  }
}

void PlanarFrame::copyInternalPlaneTo(PlanarFrame &frame, uint8_t plane)
{
  switch (plane)
  {
  case 0: if (y != NULL) BitBlt(frame.y, frame.ypitch, y, ypitch, ywidth, yheight); break;
  case 1: if (u != NULL) BitBlt(frame.u, frame.uvpitch, u, uvpitch, uvwidth, uvheight); break;
  case 2: if (v != NULL) BitBlt(frame.v, frame.uvpitch, v, uvpitch, uvwidth, uvheight); break;
  }
}

void PlanarFrame::copyChromaTo(PlanarFrame &dst)
{
  if (uvpitch != 0)
  {
    BitBlt(dst.u, dst.uvpitch, u, uvpitch, dst.uvwidth, dst.uvheight);
    BitBlt(dst.v, dst.uvpitch, v, uvpitch, dst.uvwidth, dst.uvheight);
  }
}


PlanarFrame& PlanarFrame::operator=(PlanarFrame &ob2)
{
  cpu = ob2.cpu;
  ypitch = ob2.ypitch;
  yheight = ob2.yheight;
  ywidth = ob2.ywidth;
  uvpitch = ob2.uvpitch;
  uvheight = ob2.uvheight;
  uvwidth = ob2.uvwidth;
  this->copyFrom(ob2);
  return *this;
}

#ifndef _M_X64

__declspec(align(16)) const __int64 Ymask[2] = { 0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF };

void convYUY2to422_MMX(const unsigned char *src, unsigned char *py, unsigned char *pu,
  unsigned char *pv, int pitch1, int pitch2Y, int pitch2UV, int width, int height)
{
  __asm
  {
    mov edi, src
    mov ebx, py
    mov edx, pu
    mov esi, pv
    mov ecx, width
    shr ecx, 1
    movq mm5, Ymask
    yloop :
    xor eax, eax
      align 16
      xloop :
      movq mm0, [edi + eax * 4]; VYUYVYUY
      movq mm1, [edi + eax * 4 + 8]; VYUYVYUY
      movq mm2, mm0; VYUYVYUY
      movq mm3, mm1; VYUYVYUY
      pand mm0, mm5; 0Y0Y0Y0Y
      psrlw mm2, 8; 0V0U0V0U
      pand mm1, mm5; 0Y0Y0Y0Y
      psrlw mm3, 8; 0V0U0V0U
      packuswb mm0, mm1; YYYYYYYY
      packuswb mm2, mm3; VUVUVUVU
      movq mm4, mm2; VUVUVUVU
      pand mm2, mm5; 0U0U0U0U
      psrlw mm4, 8; 0V0V0V0V
      packuswb mm2, mm2; xxxxUUUU
      packuswb mm4, mm4; xxxxVVVV
      movq[ebx + eax * 2], mm0; store y
      movd[edx + eax], mm2; store u
      movd[esi + eax], mm4; store v
      add eax, 4
      cmp eax, ecx
      jl xloop
      add edi, pitch1
      add ebx, pitch2Y
      add edx, pitch2UV
      add esi, pitch2UV
      dec height
      jnz yloop
      emms
  }
}

static void conv422toYUY2_MMX(unsigned char *py, unsigned char *pu, unsigned char *pv,
  unsigned char *dst, int pitch1Y, int pitch1UV, int pitch2, int width, int height)
{
  __asm
  {
    mov ebx, py
    mov edx, pu
    mov esi, pv
    mov edi, dst
    mov ecx, width
    shr ecx, 1
    yloop:
    xor eax, eax
      align 16
      xloop :
      movq mm0, [ebx + eax * 2]; YYYYYYYY
      movd mm1, [edx + eax]; 0000UUUU
      movd mm2, [esi + eax]; 0000VVVV
      movq mm3, mm0; YYYYYYYY
      punpcklbw mm1, mm2; VUVUVUVU
      punpcklbw mm0, mm1; VYUYVYUY
      punpckhbw mm3, mm1; VYUYVYUY
      movq[edi + eax * 4], mm0; store
      movq[edi + eax * 4 + 8], mm3; store
      add eax, 4
      cmp eax, ecx
      jl xloop
      add ebx, pitch1Y
      add edx, pitch1UV
      add esi, pitch1UV
      add edi, pitch2
      dec height
      jnz yloop
      emms
  }
}
#endif

static void convYUY2to422_SSE2_simd(const unsigned char *src, unsigned char *py, unsigned char *pu,
  unsigned char *pv, int pitch1, int pitch2Y, int pitch2UV, int width, int height)
{
  width >>= 1; // mov ecx, width
  __m128i Ymask = _mm_set1_epi16(0x00FF);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 4) {
      __m128i fullsrc = _mm_load_si128(reinterpret_cast<const __m128i *>(src + x * 4)); // VYUYVYUYVYUYVYUY
      __m128i yy = _mm_and_si128(fullsrc, Ymask); // 0Y0Y0Y0Y0Y0Y0Y0Y
      __m128i uvuv = _mm_srli_epi16(fullsrc, 8); // 0V0U0V0U0V0U0V0U
      yy = _mm_packus_epi16(yy, yy); // xxxxxxxxYYYYYYYY
      uvuv = _mm_packus_epi16(uvuv, uvuv); // xxxxxxxxVUVUVUVU
      __m128i uu = _mm_and_si128(uvuv, Ymask); // xxxxxxxx0U0U0U0U
      __m128i vv = _mm_srli_epi16(uvuv, 8); // xxxxxxxx0V0V0V0V
      uu = _mm_packus_epi16(uu, uu); // xxxxxxxxxxxxUUUU
      vv = _mm_packus_epi16(vv, vv); // xxxxxxxxxxxxVVVV
      _mm_storel_epi64(reinterpret_cast<__m128i *>(py + x * 2), yy); // store y
      *(uint32_t *)(pu + x) = _mm_cvtsi128_si32(uu); // store u
      *(uint32_t *)(pv + x) = _mm_cvtsi128_si32(vv); // store v
    }
    src += pitch1;
    py += pitch2Y;
    pu += pitch2UV;
    pv += pitch2UV;
  }
}

static void conv422toYUY2_SSE2(unsigned char *py, unsigned char *pu, unsigned char *pv,
  unsigned char *dst, int pitch1Y, int pitch1UV, int pitch2, int width, int height)
{
  width >>= 1; // mov ecx, width
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 4) {
      __m128i yy = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(py + x * 2)); // YYYYYYYY
      __m128i uu = _mm_castps_si128(_mm_load_ss(reinterpret_cast<float *>(pu + x))); // 000000000000UUUU
      __m128i vv = _mm_castps_si128(_mm_load_ss(reinterpret_cast<float *>(pv + x))); // 000000000000VVVV
      __m128i uvuv = _mm_unpacklo_epi8(uu, vv); // 00000000VUVUVUVU
      __m128i yuyv = _mm_unpacklo_epi8(yy, uvuv); // VYUYVYUYVYUYVYUY
      _mm_store_si128(reinterpret_cast<__m128i *>(dst + x * 4), yuyv);
    }
    dst += pitch2;
    py += pitch1Y;
    pu += pitch1UV;
    pv += pitch1UV;
  }
}

void PlanarFrame::convYUY2to422(const uint8_t *src, uint8_t *py, uint8_t *pu, uint8_t *pv, int pitch1, int pitch2Y, int pitch2UV,
  int width, int height)
{
  if (((cpu&CPUF_SSE2) != 0) && useSIMD && (((size_t(src) | pitch1) & 15) == 0))
    convYUY2to422_SSE2_simd(src, py, pu, pv, pitch1, pitch2Y, pitch2UV, width, height);
  else
  {
#ifndef _M_X64
    if (((cpu&CPUF_MMX) != 0) && useSIMD)
      convYUY2to422_MMX(src, py, pu, pv, pitch1, pitch2Y, pitch2UV, width, height);
    else
#endif
    {
      width >>= 1;
      for (int y = 0; y < height; ++y)
      {
        int x_1 = 0, x_2 = 0;

        for (int x = 0; x < width; ++x)
        {
          py[x_1] = src[x_2];
          pu[x] = src[x_2 + 1];
          py[x_1 + 1] = src[x_2 + 2];
          pv[x] = src[x_2 + 3];
          x_1 += 2;
          x_2 += 4;
        }
        py += pitch2Y;
        pu += pitch2UV;
        pv += pitch2UV;
        src += pitch1;
      }
    }
  }
}


void PlanarFrame::conv422toYUY2(uint8_t *py, uint8_t *pu, uint8_t *pv, uint8_t *dst, int pitch1Y, int pitch1UV, int pitch2,
  int width, int height)
{
  if (((cpu&CPUF_SSE2) != 0) && useSIMD && ((size_t(dst) & 15) == 0))
    conv422toYUY2_SSE2(py, pu, pv, dst, pitch1Y, pitch1UV, pitch2, width, height);
  else
  {
#ifndef _M_X64
    if (((cpu&CPUF_MMX) != 0) && useSIMD) 
      conv422toYUY2_MMX(py, pu, pv, dst, pitch1Y, pitch1UV, pitch2, width, height);
    else
#endif
    {
      width >>= 1;
      for (int y = 0; y < height; ++y)
      {
        int x_1 = 0, x_2 = 0;

        for (int x = 0; x < width; ++x)
        {
          dst[x_2] = py[x_1];
          dst[x_2 + 1] = pu[x];
          dst[x_2 + 2] = py[x_1 + 1];
          dst[x_2 + 3] = pv[x];
          x_1 += 2;
          x_2 += 4;
        }
        py += pitch1Y;
        pu += pitch1UV;
        pv += pitch1UV;
        dst += pitch2;
      }
    }
  }
}


void PlanarFrame::convRGB24to444(const uint8_t *src, uint8_t *py, uint8_t *pu, uint8_t *pv, int pitch1, int pitch2Y, int pitch2UV,
  int width, int height)
{
  for (int y = 0; y < height; ++y)
  {
    int x_3 = 0;

    for (int x = 0; x < width; ++x)
    {
      py[x] = src[x_3];
      pu[x] = src[x_3 + 1];
      pv[x] = src[x_3 + 2];
      x_3 += 3;
    }
    src += pitch1;
    py += pitch2Y;
    pu += pitch2UV;
    pv += pitch2UV;
  }
}

void PlanarFrame::conv444toRGB24(uint8_t *py, uint8_t *pu, uint8_t *pv, uint8_t *dst, int pitch1Y, int pitch1UV, int pitch2,
  int width, int height)
{
  dst += (height - 1)*pitch2;
  for (int y = 0; y < height; ++y)
  {
    int x_3 = 0;

    for (int x = 0; x < width; ++x)
    {
      dst[x_3] = py[x];
      dst[x_3 + 1] = pu[x];
      dst[x_3 + 2] = pv[x];
      x_3 += 3;
    }
    py += pitch1Y;
    pu += pitch1UV;
    pv += pitch1UV;
    dst -= pitch2;
  }
}

// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.

// from AviSynth 2.55 source...
// copied so we don't need an
// IScriptEnvironment pointer 
// to call it

void PlanarFrame::BitBlt(uint8_t *dstp, int dst_pitch, const uint8_t *srcp, int src_pitch, int row_size, int height)
{
  if ((height == 0) || (row_size == 0) || (dst_pitch == 0) || (src_pitch == 0)) return;

  if ((height == 1) || ((dst_pitch == src_pitch) && (src_pitch == row_size))) memcpy(dstp, srcp, src_pitch*height);
  else
  {
    for (int y = height; y > 0; --y)
    {
      memcpy(dstp, srcp, row_size);
      dstp += dst_pitch;
      srcp += src_pitch;
    }
  }
}
