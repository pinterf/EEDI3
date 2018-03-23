/*
**   eedi3 (enhanced edge directed interpolation 3). Works by finding the
**   best non-decreasing (non-crossing) warping between two lines according to
**   a cost functional. Doesn't really have anything to do with eedi2 aside
**   from doing edge-directed interpolation (they use different techniques).
**
**   Copyright (C) 2015 Shane Panke
**
**   Copyright (C) 2010 Kevin Stone
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

#include "eedi3.h"

eedi3::eedi3(PClip _child, int _field, bool _dh, bool _Y, bool _U, bool _V, float _alpha,
  float _beta, float _gamma, int _nrad, int _mdis, bool _hp, bool _ucubic, bool _cost3,
  int _vcheck, float _vthresh0, float _vthresh1, float _vthresh2, PClip _sclip, int _threads,
  PClip _mclip, int opt, IScriptEnvironment *env)
  : GenericVideoFilter(_child), field(_field), dh(_dh), Y(_Y), U(_U),
  V(_V), alpha(_alpha), beta(_beta), gamma(_gamma), nrad(_nrad), mdis(_mdis), hp(_hp),
  ucubic(_ucubic), cost3(_cost3), vcheck(_vcheck), vthresh0(_vthresh0), vthresh1(_vthresh1),
  vthresh2(_vthresh2), sclip(_sclip), mclip(_mclip), _sse2_flag(false)
{
  if (field < -2 || field > 3)
    env->ThrowError("eedi3:  field must be set to -2, -1, 0, 1, 2, or 3!");
  if (dh && (field < -1 || field > 1))
    env->ThrowError("eedi3:  field must be set to -1, 0, or 1 when dh=true!");
  if (alpha < 0.0f || alpha > 1.0f)
    env->ThrowError("eedi3:  0 <= alpha <= 1!\n");
  if (beta < 0.0f || beta > 1.0f)
    env->ThrowError("eedi3:  0 <= beta <= 1!\n");
  if (alpha + beta > 1.0f)
    env->ThrowError("eedi3:  0 <= alpha+beta <= 1!\n");
  if (gamma < 0.0f)
    env->ThrowError("eedi3:  0 <= gamma!\n");
  if (nrad < 0 || nrad > 3)
    env->ThrowError("eedi3:  0 <= nrad <= 3!\n");
  if (mdis < 1 || mdis > 40)
    env->ThrowError("eedi3:  1 <= mdis <= 40!\n");
  if (vcheck < 0 || vcheck > 3)
    env->ThrowError("eedi3:  0 <= vcheck <= 3!\n");
  if (vcheck > 0 && (vthresh0 <= 0.0f || vthresh1 <= 0.0f || vthresh2 <= 0.0f))
    env->ThrowError("eedi3:  0 < vthresh0 , 0 < vthresh1 , 0 < vthresh2!\n");
  if (field == -2)
    field = child->GetParity(0) ? 3 : 2;
  else if (field == -1)
    field = child->GetParity(0) ? 1 : 0;

  if (mclip)
  {
    const ::VideoInfo &	vi2 = mclip->GetVideoInfo();
    if (vi.height != vi2.height
      || vi.width != vi2.width
      || vi.num_frames != vi2.num_frames
      || !vi.IsSameColorspace(vi2))
    {
      env->ThrowError("eedi3:  mclip doesn't match source clip!\n");
    }
  }

  if (opt == 2)
  {
    _sse2_flag = true;
  }
  else if (opt != 1)
  {
    _sse2_flag = ((env->GetCPUFlags() & CPUF_SSE2) != 0);
  }
  _sse2_flag = (_sse2_flag && !hp);	// Half-pel SSE2 not implemented yet

  if (field > 1)
  {
    vi.num_frames *= 2;
    vi.SetFPS(vi.fps_numerator * 2, vi.fps_denominator);
  }
  if (dh)
    vi.height *= 2;
  vi.SetFieldBased(false);
  child->SetCacheHints(CACHE_GET_RANGE, 3);
  mcpPF = 0;

  const int cpuFlags = env->GetCPUFlags();

  if (mclip)
  {
    ::VideoInfo	vi2 = vi;
    vi2.height /= 2;
    mcpPF = new PlanarFrame(vi2, cpuFlags);
  }
  srcPF = new PlanarFrame(cpuFlags);
  srcPF->createPlanar(vi.height + MARGIN_V * 2, (vi.IsYV12() ? (vi.height >> 1) : vi.height) + MARGIN_V * 2,
    vi.width + MARGIN_H * 2, (vi.IsRGB24() ? vi.width : (vi.width >> 1)) + MARGIN_H * 2);
  dstPF = new PlanarFrame(vi, cpuFlags);
  scpPF = new PlanarFrame(vi, cpuFlags);
  if (_threads > 0)
    omp_set_num_threads(_threads);
  const int nthreads = omp_get_max_threads();
  workspace = (uint8_t**)calloc(nthreads, sizeof(*workspace));
  dmapa = (int16_t*)_aligned_malloc(dstPF->GetPitch(0)*dstPF->GetHeight(0) * sizeof(*dmapa), 16);
  if (!workspace || !dmapa)
    env->ThrowError("eedi3:  malloc failure!\n");
  const int tpitch = max(mdis * ((hp) ? 4 : 2) + 1, 16);
  int workspace_size = vi.width * tpitch * 4 * sizeof(float);
  if (_sse2_flag)
  {
    workspace_size = (vi.width + 2 * Eedi3Sse::MARGIN_H) * 4 * sizeof(uint16_t) * Eedi3Sse::COL_H; // src
    workspace_size += vi.width * 2 * sizeof(int16_t) * Eedi3Sse::COL_H; // dst + dmap
    workspace_size += (vi.width * sizeof(uint8_t) * Eedi3Sse::COL_H + 15) & -16; // mask
    workspace_size += vi.width * tpitch * 5 * sizeof(float) * Eedi3Sse::VECTSIZE; // temp
  }
  for (int i = 0; i < nthreads; ++i)
  {
    workspace[i] = (uint8_t*)_aligned_malloc(workspace_size, 16);
    if (!workspace[i])
      env->ThrowError("eedi3:  malloc failure!\n");
  }
  if (vcheck > 0 && sclip)
  {
    VideoInfo vi2 = sclip->GetVideoInfo();
    if (vi.height != vi2.height ||
      vi.width != vi2.width ||
      vi.num_frames != vi2.num_frames ||
      !vi.IsSameColorspace(vi2))
      env->ThrowError("eedi3:  sclip doesn't match!\n");
  }
}

eedi3::~eedi3()
{
  delete srcPF;
  delete dstPF;
  delete scpPF;
  delete mcpPF;
  const int nthreads = omp_get_num_threads();
  for (int i = 0; i < nthreads; ++i)
    _aligned_free(workspace[i]);
  free(workspace);
  _aligned_free(dmapa);
}

void expand_mask(bool bmask[], const uint8_t maskp[], int width, int mdis)
{
  const int	minmdis = (width < mdis) ? width : mdis;

  int			last = -666999;

  for (int x = 0; x < minmdis; ++x)
  {
    if (maskp[x] != 0)
    {
      last = x + mdis;
    }
  }

  for (int x = 0; x < width - minmdis; ++x)
  {
    if (maskp[x + mdis] != 0)
    {
      last = x + mdis * 2;
    }
    bmask[x] = (x <= last);
  }

  for (int x = width - minmdis; x < width; ++x)
  {
    bmask[x] = (x <= last);
  }
}

// Full-pel steps
void interpLineFP(const uint8_t *srcp, const int width, const int pitch,
  const float alpha, const float beta, const float gamma, const int nrad,
  const int mdis, float *temp, uint8_t *dstp, int16_t *dmap, const bool ucubic,
  const bool cost3, const uint8_t *maskp)
{
  const uint8_t *src3p = srcp - 3 * pitch;
  const uint8_t *src1p = srcp - 1 * pitch;
  const uint8_t *src1n = srcp + 1 * pitch;
  const uint8_t *src3n = srcp + 3 * pitch;
  const int tpitch = mdis * 2 + 1;
  float *ccosts = temp;	// Array of mdis*2+1 costs for each pixel of the line
  float *pcosts = ccosts + width * tpitch;
  int *pbackt = (int*)(pcosts + width * tpitch);
  int *fpath = pbackt + width * tpitch;
  bool *bmask = (bool *)(fpath + width);
  if (maskp != 0)
  {
    memset(ccosts, 0, sizeof(ccosts[0]) * tpitch * width);
    expand_mask(bmask, maskp, width, mdis);
  }
  // calculate all connection costs
  if (!cost3)
  {
    for (int x = 0; x < width; ++x)
    {
      if (maskp == 0 || bmask[x])
      {
        const int umax = min(min(x, width - 1 - x), mdis);
        for (int u = -umax; u <= umax; ++u)
        {
          int s = 0;
          for (int k = -nrad; k <= nrad; ++k)
            s +=
            abs(src3p[x + u + k] - src1p[x - u + k]) +
            abs(src1p[x + u + k] - src1n[x - u + k]) +
            abs(src1n[x + u + k] - src3n[x - u + k]);
          const int ip = (src1p[x + u] + src1n[x - u] + 1) >> 1; // should use cubic if ucubic=true
          const int v = abs(src1p[x] - ip) + abs(src1n[x] - ip);
          ccosts[x*tpitch + mdis + u] = alpha * s + beta * abs(u) + (1.0f - alpha - beta)*v;
        }
      }
    }
  }
  else
  {
    for (int x = 0; x < width; ++x)
    {
      if (maskp == 0 || bmask[x])
      {
        const int umax = min(min(x, width - 1 - x), mdis);
        for (int u = -umax; u <= umax; ++u)
        {
          int s0 = 0, s1 = -1, s2 = -1;
          for (int k = -nrad; k <= nrad; ++k)
            s0 +=
            abs(src3p[x + u + k] - src1p[x - u + k]) +
            abs(src1p[x + u + k] - src1n[x - u + k]) +
            abs(src1n[x + u + k] - src3n[x - u + k]);
          if ((u >= 0 && x >= u * 2) || (u <= 0 && x < width + u * 2))
          {
            s1 = 0;
            for (int k = -nrad; k <= nrad; ++k)
              s1 +=
              abs(src3p[x + k] - src1p[x - u * 2 + k]) +
              abs(src1p[x + k] - src1n[x - u * 2 + k]) +
              abs(src1n[x + k] - src3n[x - u * 2 + k]);
          }
          if ((u <= 0 && x >= -u * 2) || (u >= 0 && x < width + u * 2)) // LDS: fixed u -> -u
          {
            s2 = 0;
            for (int k = -nrad; k <= nrad; ++k)
              s2 +=
              abs(src3p[x + u * 2 + k] - src1p[x + k]) +
              abs(src1p[x + u * 2 + k] - src1n[x + k]) +
              abs(src1n[x + u * 2 + k] - src3n[x + k]);
          }
          s1 = s1 >= 0 ? s1 : (s2 >= 0 ? s2 : s0);
          s2 = s2 >= 0 ? s2 : (s1 >= 0 ? s1 : s0);
          const int ip = (src1p[x + u] + src1n[x - u] + 1) >> 1; // should use cubic if ucubic=true
          const int v = abs(src1p[x] - ip) + abs(src1n[x] - ip);
          ccosts[x*tpitch + mdis + u] = alpha * (s0 + s1 + s2)*0.333333f + beta * abs(u) + (1.0f - alpha - beta)*v;
        }
      }
    }
  }
  // calculate path costs
  pcosts[mdis] = ccosts[mdis];
  for (int x = 1; x < width; ++x)
  {
    float *tT = ccosts + x * tpitch;
    float *ppT = pcosts + (x - 1)*tpitch;
    float *pT = pcosts + x * tpitch;
    int   *piT = pbackt + (x - 1)*tpitch;
    if (maskp != 0 && !bmask[x])
    {
      if (x == 1)
      {
        const int umax = min(min(x, width - 1 - x), mdis);
        for (int u = -umax; u <= umax; ++u)
        {
          pT[mdis + u] = tT[mdis + u];
        }
        memset(piT, 0, sizeof(*piT) * tpitch);
      }
      else
      {
        memcpy(pT, ppT, sizeof(*ppT) * tpitch);
        memcpy(piT, piT - tpitch, sizeof(*piT) * tpitch);
        const int pumax = min(x - 1, width - x);
        if (pumax < mdis)
        {
          piT[mdis - pumax] = 1 - pumax;
          piT[mdis + pumax] = pumax - 1;
        }
      }
    }
    else
    {
      const int umax = min(min(x, width - 1 - x), mdis);
      for (int u = -umax; u <= umax; ++u)
      {
        int idx;
        float bval = FLT_MAX;
        const int umax2 = min(min(x - 1, width - x), mdis);
        for (int v = max(-umax2, u - 1); v <= min(umax2, u + 1); ++v)
        {
          const double y = ppT[mdis + v] + gamma * abs(u - v);
          const float ccost = (float)min(y, FLT_MAX*0.9);
          if (ccost < bval)
          {
            bval = ccost;
            idx = v;
          }
        }
        const double y = bval + tT[mdis + u];
        pT[mdis + u] = (float)min(y, FLT_MAX*0.9);
        piT[mdis + u] = idx;
      }
    }
  }
  // backtrack
  fpath[width - 1] = 0;
  for (int x = width - 2; x >= 0; --x)
    fpath[x] = pbackt[x*tpitch + mdis + fpath[x + 1]];
  // interpolate
  for (int x = 0; x < width; ++x)
  {
    if (maskp != 0 && !bmask[x])
    {
      dmap[x] = 0;
      if (ucubic)
      {
        dstp[x] = min(max((9 * (src1p[x] + src1n[x]) -
          (src3p[x] + src3n[x]) + 8) >> 4, 0), 255);
      }
      else
      {
        dstp[x] = (src1p[x] + src1n[x] + 1) >> 1;
      }
    }
    else
    {
      const int dir = fpath[x];
      dmap[x] = dir;
      const int ad = abs(dir);
      if (ucubic && x >= ad * 3 && x <= width - 1 - ad * 3)
        dstp[x] = min(max((9 * (src1p[x + dir] + src1n[x - dir]) -
        (src3p[x + dir * 3] + src3n[x - dir * 3]) + 8) >> 4, 0), 255);
      else
        dstp[x] = (src1p[x + dir] + src1n[x - dir] + 1) >> 1;
    }
  }
}

// Half-pel steps
void interpLineHP(const uint8_t *srcp, const int width, const int pitch,
  const float alpha, const float beta, const float gamma, const int nrad,
  const int mdis, float *temp, uint8_t *dstp, int16_t *dmap, const bool ucubic,
  const bool cost3, const uint8_t *maskp)
{
  const uint8_t *src3p = srcp - 3 * pitch;
  const uint8_t *src1p = srcp - 1 * pitch;
  const uint8_t *src1n = srcp + 1 * pitch;
  const uint8_t *src3n = srcp + 3 * pitch;
  const int tpitch = mdis * 4 + 1;
  float *ccosts = temp;
  float *pcosts = ccosts + width * tpitch;
  int *pbackt = (int*)(pcosts + width * tpitch);
  int *fpath = pbackt + width * tpitch;
  uint8_t *hp3p = (uint8_t*)fpath;
  uint8_t *hp1p = hp3p + width;
  uint8_t *hp1n = hp1p + width;
  uint8_t *hp3n = hp1n + width;
  bool *bmask = (bool *)(hp3n + width);
  // calculate half pel values
  for (int x = 0; x < width - 1; ++x)
  {
    if (!ucubic || (x == 0 || x == width - 2))
    {
      hp3p[x] = (src3p[x] + src3p[x + 1] + 1) >> 1;
      hp1p[x] = (src1p[x] + src1p[x + 1] + 1) >> 1;
      hp1n[x] = (src1n[x] + src1n[x + 1] + 1) >> 1;
      hp3n[x] = (src3n[x] + src3n[x + 1] + 1) >> 1;
    }
    else
    {
      hp3p[x] = min(max((9 * (src3p[x] + src3p[x + 1]) - (src3p[x - 1] + src3p[x + 2]) + 8) >> 4, 0), 255);
      hp1p[x] = min(max((9 * (src1p[x] + src1p[x + 1]) - (src1p[x - 1] + src1p[x + 2]) + 8) >> 4, 0), 255);
      hp1n[x] = min(max((9 * (src1n[x] + src1n[x + 1]) - (src1n[x - 1] + src1n[x + 2]) + 8) >> 4, 0), 255);
      hp3n[x] = min(max((9 * (src3n[x] + src3n[x + 1]) - (src3n[x - 1] + src3n[x + 2]) + 8) >> 4, 0), 255);
    }
  }
  if (maskp != 0)
  {
    memset(ccosts, 0, sizeof(ccosts[0]) * tpitch * width);
    expand_mask(bmask, maskp, width, mdis);
  }
  // calculate all connection costs
  if (!cost3)
  {
    for (int x = 0; x < width; ++x)
    {
      if (maskp == 0 || bmask[x])
      {
        const int umax = min(min(x, width - 1 - x), mdis);
        for (int u = -umax * 2; u <= umax * 2; ++u)
        {
          int s = 0, ip;
          const int u2 = u >> 1;
          if (!(u & 1))
          {
            for (int k = -nrad; k <= nrad; ++k)
              s +=
              abs(src3p[x + u2 + k] - src1p[x - u2 + k]) +
              abs(src1p[x + u2 + k] - src1n[x - u2 + k]) +
              abs(src1n[x + u2 + k] - src3n[x - u2 + k]);
            ip = (src1p[x + u2] + src1n[x - u2] + 1) >> 1; // should use cubic if ucubic=true
          }
          else
          {
            for (int k = -nrad; k <= nrad; ++k)
              s +=
              abs(hp3p[x + u2 + k] - hp1p[x - u2 - 1 + k]) +
              abs(hp1p[x + u2 + k] - hp1n[x - u2 - 1 + k]) +
              abs(hp1n[x + u2 + k] - hp3n[x - u2 - 1 + k]);
            ip = (hp1p[x + u2] + hp1n[x - u2 - 1] + 1) >> 1; // should use cubic if ucubic=true
          }
          const int v = abs(src1p[x] - ip) + abs(src1n[x] - ip);
          ccosts[x*tpitch + mdis * 2 + u] = alpha * s + beta * abs(u)*0.5f + (1.0f - alpha - beta)*v;
        }
      }
    }
  }
  else
  {
    for (int x = 0; x < width; ++x)
    {
      if (maskp == 0 || bmask[x])
      {
        const int umax = min(min(x, width - 1 - x), mdis);
        for (int u = -umax * 2; u <= umax * 2; ++u)
        {
          int s0 = 0, s1 = -1, s2 = -1, ip;
          const int u2 = u >> 1;
          if (!(u & 1))
          {
            for (int k = -nrad; k <= nrad; ++k)
              s0 +=
              abs(src3p[x + u2 + k] - src1p[x - u2 + k]) +
              abs(src1p[x + u2 + k] - src1n[x - u2 + k]) +
              abs(src1n[x + u2 + k] - src3n[x - u2 + k]);
            ip = (src1p[x + u2] + src1n[x - u2] + 1) >> 1; // should use cubic if ucubic=true
          }
          else
          {
            for (int k = -nrad; k <= nrad; ++k)
              s0 +=
              abs(hp3p[x + u2 + k] - hp1p[x - u2 - 1 + k]) +
              abs(hp1p[x + u2 + k] - hp1n[x - u2 - 1 + k]) +
              abs(hp1n[x + u2 + k] - hp3n[x - u2 - 1 + k]);
            ip = (hp1p[x + u2] + hp1n[x - u2 - 1] + 1) >> 1; // should use cubic if ucubic=true
          }
          if ((u >= 0 && x >= u) || (u <= 0 && x < width + u))
          {
            s1 = 0;
            for (int k = -nrad; k <= nrad; ++k)
              s1 +=
              abs(src3p[x + k] - src1p[x - u + k]) +
              abs(src1p[x + k] - src1n[x - u + k]) +
              abs(src1n[x + k] - src3n[x - u + k]);
          }
          if ((u <= 0 && x >= -u) || (u >= 0 && x < width + u)) // LDS: fixed u -> -u
          {
            s2 = 0;
            for (int k = -nrad; k <= nrad; ++k)
              s2 +=
              abs(src3p[x + u + k] - src1p[x + k]) +
              abs(src1p[x + u + k] - src1n[x + k]) +
              abs(src1n[x + u + k] - src3n[x + k]);
          }
          s1 = s1 >= 0 ? s1 : (s2 >= 0 ? s2 : s0);
          s2 = s2 >= 0 ? s2 : (s1 >= 0 ? s1 : s0);
          const int v = abs(src1p[x] - ip) + abs(src1n[x] - ip);
          ccosts[x*tpitch + mdis * 2 + u] = alpha * (s0 + s1 + s2)*0.333333f + beta * abs(u)*0.5f + (1.0f - alpha - beta)*v;
        }
      }
    }
  }
  // calculate path costs
  pcosts[mdis * 2] = ccosts[mdis * 2];
  for (int x = 1; x < width; ++x)
  {
    float *tT = ccosts + x * tpitch;
    float *ppT = pcosts + (x - 1)*tpitch;
    float *pT = pcosts + x * tpitch;
    int *piT = pbackt + (x - 1)*tpitch;
    if (maskp != 0 && !bmask[x])
    {
      if (x == 1)
      {
        const int umax = min(min(x, width - 1 - x), mdis);
        for (int u = -umax * 2; u <= umax * 2; ++u)
        {
          pT[mdis * 2 + u] = tT[mdis * 2 + u];
        }
        memset(piT, 0, sizeof(*piT) * tpitch);
      }
      else
      {
        memcpy(pT, ppT, sizeof(*ppT) * tpitch);
        memcpy(piT, piT - tpitch, sizeof(*piT) * tpitch);
        const int pumax = min(x - 1, width - x);
        if (pumax < mdis)
        {
          piT[mdis - pumax * 2] = (1 - pumax) * 2;
          piT[mdis - pumax * 2 + 1] = (1 - pumax) * 2;
          piT[mdis + pumax * 2 - 1] = (pumax - 1) * 2;
          piT[mdis + pumax * 2] = (pumax - 1) * 2;
        }
      }
    }
    else
    {
      const int umax = min(min(x, width - 1 - x), mdis);
      for (int u = -umax * 2; u <= umax * 2; ++u)
      {
        int idx;
        float bval = FLT_MAX;
        const int umax2 = min(min(x - 1, width - x), mdis);
        for (int v = max(-umax2 * 2, u - 2); v <= min(umax2 * 2, u + 2); ++v)
        {
          const double y = ppT[mdis * 2 + v] + gamma * abs(u - v)*0.5f;
          const float ccost = (float)min(y, FLT_MAX*0.9);
          if (ccost < bval)
          {
            bval = ccost;
            idx = v;
          }
        }
        const double y = bval + tT[mdis * 2 + u];
        pT[mdis * 2 + u] = (float)min(y, FLT_MAX*0.9);
        piT[mdis * 2 + u] = idx;
      }
    }
  }
  // backtrack
  fpath[width - 1] = 0;
  for (int x = width - 2; x >= 0; --x)
    fpath[x] = pbackt[x*tpitch + mdis * 2 + fpath[x + 1]];
  // interpolate
  for (int x = 0; x < width; ++x)
  {
    if (maskp != 0 && !bmask[x])
    {
      dmap[x] = 0;
      if (ucubic)
        dstp[x] = min(max((9 * (src1p[x] + src1n[x]) -
        (src3p[x] + src3n[x]) + 8) >> 4, 0), 255);
      else
        dstp[x] = (src1p[x] + src1n[x] + 1) >> 1;
    }
    else
    {
      const int dir = fpath[x];
      dmap[x] = dir;
      if (!(dir & 1))
      {
        const int d2 = dir >> 1;
        const int ad = abs(d2);
        if (ucubic && x >= ad * 3 && x <= width - 1 - ad * 3)
          dstp[x] = min(max((9 * (src1p[x + d2] + src1n[x - d2]) -
          (src3p[x + d2 * 3] + src3n[x - d2 * 3]) + 8) >> 4, 0), 255);
        else
          dstp[x] = (src1p[x + d2] + src1n[x - d2] + 1) >> 1;
      }
      else
      {
        const int d20 = dir >> 1;
        const int d21 = (dir + 1) >> 1;
        const int d30 = (dir * 3) >> 1;
        const int d31 = (dir * 3 + 1) >> 1;
        const int ad = max(abs(d30), abs(d31));
        if (ucubic && x >= ad && x <= width - 1 - ad)
        {
          const int c0 = src3p[x + d30] + src3p[x + d31];
          const int c1 = src1p[x + d20] + src1p[x + d21]; // should use cubic if ucubic=true
          const int c2 = src1n[x - d20] + src1n[x - d21]; // should use cubic if ucubic=true
          const int c3 = src3n[x - d30] + src3n[x - d31];
          dstp[x] = min(max((9 * (c1 + c2) - (c0 + c3) + 16) >> 5, 0), 255);
        }
        else
          dstp[x] = (src1p[x + d20] + src1p[x + d21] + src1n[x - d20] + src1n[x - d21] + 2) >> 2;
      }
    }
  }
}

PVideoFrame __stdcall eedi3::GetFrame(int n, IScriptEnvironment *env)
{
  int field_n;
  int field_s = n;
  if (field > 1)
  {
    if (n & 1) field_n = field == 3 ? 0 : 1;
    else field_n = field == 3 ? 1 : 0;
    field_s >>= 1;
  }
  else
    field_n = field;
  copyPad(field_s, field_n, env);
  if (mclip)
  {
    copyMask(field_s, field_n, env);
  }
  if (vcheck > 0 && sclip)
    scpPF->copyFrom(sclip->GetFrame(n, env), vi);
  for (int b = 0; b < 3; ++b)
  {
    if ((b == 0 && !Y) ||
      (b == 1 && !U) ||
      (b == 2 && !V))
      continue;
    const uint8_t *srcp = srcPF->GetPtr(b);
    const int spitch = srcPF->GetPitch(b);
    const int width = srcPF->GetWidth(b);
    const int height = srcPF->GetHeight(b);
    uint8_t *dstp = dstPF->GetPtr(b);
    const int dpitch = dstPF->GetPitch(b);
    env->BitBlt(dstp + (1 - field_n)*dpitch,
      dpitch * 2, srcp + (MARGIN_V + 1 - field_n)*spitch + MARGIN_H,
      spitch * 2, width - MARGIN_H * 2, (height - MARGIN_V * 2) >> 1);
    uint8_t *   maskp_base = 0;
    int               mpitch = 0;
    if (mclip)
    {
      maskp_base = mcpPF->GetPtr(b);
      mpitch = mcpPF->GetPitch(b);
    }

    // SSE2
    if (_sse2_flag)
    {
      assert(!hp);

      srcp += MARGIN_V * spitch;
      dstp += field_n * dpitch;

      const int   plane_w = width - MARGIN_H * 2;
      const int   plane_h = height - MARGIN_V * 2;
      const int   plane_hs = (plane_h + field_n) >> 1; // Number of existing source lines
      const int   plane_hi = plane_h - plane_hs;       // Number of interpolated lines
      const int   packedline_stride_pix = plane_w + 2 * Eedi3Sse::MARGIN_H;
      const int   packedline_stride =
        packedline_stride_pix * sizeof(uint16_t) * Eedi3Sse::COL_H;

      // ~99% of the processing time is spent in this loop
#pragma omp parallel for
      for (int y = field_n; y < plane_h; y += 2 * Eedi3Sse::COL_H)
      {
        const int      tidx = omp_get_thread_num();
        const int      off = (y - field_n) >> 1;
        uint8_t* maskp = 0;
        uint8_t *      src_ptr = workspace[tidx];
        uint8_t *      dst_ptr = src_ptr + 4 * packedline_stride;
        uint8_t *      dma_ptr = dst_ptr + plane_w * Eedi3Sse::COL_H * sizeof(uint16_t);
        uint8_t *      msk_ptr = dma_ptr + ((plane_w * Eedi3Sse::COL_H * sizeof(uint8_t) + 15) & -16);
        uint8_t *      tmp_ptr = msk_ptr + plane_w * Eedi3Sse::COL_H * sizeof(int16_t);
        if (maskp_base == 0)
        {
          msk_ptr = 0;
        }
        else
        {
          Eedi3Sse::prepare_mask_8bits(
            msk_ptr,
            maskp_base,
            mpitch,
            1,
            plane_w,
            plane_hs,
            off
          );
        }
        Eedi3Sse::prepare_lines_8bits(
          reinterpret_cast <uint16_t *> (src_ptr),
          packedline_stride_pix,
          srcp + spitch * (1 - field_n) + MARGIN_H,  // +spitch* because the C++ version points on the interpolated line. We need the next one.
          spitch * 2,
          1,
          plane_w,
          plane_hs,
          off + field_n
        );
        Eedi3Sse::interp_lines_full_pel(
          reinterpret_cast <const __m128i *> (src_ptr),
          reinterpret_cast <__m128i *> (dst_ptr),
          msk_ptr,
          tmp_ptr,
          reinterpret_cast <__m128i *> (dma_ptr),
          plane_w,
          packedline_stride_pix,
          alpha, beta, gamma,
          nrad, mdis, ucubic, cost3
        );
        Eedi3Sse::copy_result_lines_8bits(
          dstp,
          dpitch * 2,
          reinterpret_cast <const uint16_t *> (dst_ptr),
          plane_w,
          1,
          plane_w,
          plane_hi,
          off
        );
        if (vcheck > 0)
        {
          Eedi3Sse::copy_result_dmap(
            dmapa,
            dpitch,
            reinterpret_cast <int16_t *> (dma_ptr),
            plane_w,
            plane_w,
            plane_hi,
            off
          );
        }
      }

      srcp += field_n * spitch;
    }

    // C++ only
    else
    {
      srcp += (MARGIN_V + field_n)*spitch;
      dstp += field_n * dpitch;

      // ~99% of the processing time is spent in this loop
#pragma omp parallel for
      for (int y = MARGIN_V + field_n; y < height - MARGIN_V; y += 2)
      {
        const int tidx = omp_get_thread_num();
        const int off = (y - MARGIN_V - field_n) >> 1;
        uint8_t* maskp = 0;
        if (maskp_base != 0)
        {
          maskp = maskp_base + mpitch * off;
        }
        if (hp)
          interpLineHP(srcp + MARGIN_H + off * 2 * spitch, width - MARGIN_H * 2, spitch, alpha, beta,
            gamma, nrad, mdis, (float*)(workspace[tidx]), dstp + off * 2 * dpitch,
            dmapa + off * dpitch, ucubic, cost3, maskp);
        else
          interpLineFP(srcp + MARGIN_H + off * 2 * spitch, width - MARGIN_H * 2, spitch, alpha, beta,
            gamma, nrad, mdis, (float*)(workspace[tidx]), dstp + off * 2 * dpitch,
            dmapa + off * dpitch, ucubic, cost3, maskp);
      }
    }
    if (vcheck > 0)
    {
      int16_t *dstpd = dmapa;
      const uint8_t *scpp = NULL;
      int scpitch;
      if (sclip)
      {
        scpitch = scpPF->GetPitch(b);
        scpp = scpPF->GetPtr(b) + field_n * scpitch;
      }
      for (int y = MARGIN_V + field_n; y < height - MARGIN_V; y += 2)
      {
        if (y >= 6 && y < height - 6)
        {
          const uint8_t *dst3p = srcp - 3 * spitch + MARGIN_H;
          const uint8_t *dst2p = dstp - 2 * dpitch;
          const uint8_t *dst1p = dstp - 1 * dpitch;
          const uint8_t *dst1n = dstp + 1 * dpitch;
          const uint8_t *dst2n = dstp + 2 * dpitch;
          const uint8_t *dst3n = srcp + 3 * spitch + MARGIN_H;
          uint8_t *tline = workspace[0];
          for (int x = 0; x < width - MARGIN_H * 2; ++x)
          {
            const int dirc = dstpd[x];
            const int cint = scpp ? scpp[x] :
              min(max((9 * (dst1p[x] + dst1n[x]) - (dst3p[x] + dst3n[x]) + 8) >> 4, 0), 255);
            if (dirc == 0)
            {
              tline[x] = cint;
              continue;
            }
            const int dirt = dstpd[x - dpitch];
            const int dirb = dstpd[x + dpitch];
            if (max(dirc*dirt, dirc*dirb) < 0 || (dirt == dirb && dirt == 0))
            {
              tline[x] = cint;
              continue;
            }
            int it, ib, vt, vb, vc;
            vc = abs(dstp[x] - dst1p[x]) + abs(dstp[x] - dst1n[x]);
            if (hp)
            {
              if (!(dirc & 1))
              {
                const int d2 = dirc >> 1;
                it = (dst2p[x + d2] + dstp[x - d2] + 1) >> 1;
                vt = abs(dst2p[x + d2] - dst1p[x + d2]) + abs(dstp[x + d2] - dst1p[x + d2]);
                ib = (dstp[x + d2] + dst2n[x - d2] + 1) >> 1;
                vb = abs(dst2n[x - d2] - dst1n[x - d2]) + abs(dstp[x - d2] - dst1n[x - d2]);
              }
              else
              {
                const int d20 = dirc >> 1;
                const int d21 = (dirc + 1) >> 1;
                const int pa2p = dst2p[x + d20] + dst2p[x + d21] + 1;
                const int pa1p = dst1p[x + d20] + dst1p[x + d21] + 1;
                const int ps0 = dstp[x - d20] + dstp[x - d21] + 1;
                const int pa0 = dstp[x + d20] + dstp[x + d21] + 1;
                const int ps1n = dst1n[x - d20] + dst1n[x - d21] + 1;
                const int ps2n = dst2n[x - d20] + dst2n[x - d21] + 1;
                it = (pa2p + ps0) >> 2;
                vt = (abs(pa2p - pa1p) + abs(pa0 - pa1p)) >> 1;
                ib = (pa0 + ps2n) >> 2;
                vb = (abs(ps2n - ps1n) + abs(ps0 - ps1n)) >> 1;
              }
            }
            else
            {
              it = (dst2p[x + dirc] + dstp[x - dirc] + 1) >> 1;
              vt = abs(dst2p[x + dirc] - dst1p[x + dirc]) + abs(dstp[x + dirc] - dst1p[x + dirc]);
              ib = (dstp[x + dirc] + dst2n[x - dirc] + 1) >> 1;
              vb = abs(dst2n[x - dirc] - dst1n[x - dirc]) + abs(dstp[x - dirc] - dst1n[x - dirc]);
            }
            const int d0 = abs(it - dst1p[x]);
            const int d1 = abs(ib - dst1n[x]);
            const int d2 = abs(vt - vc);
            const int d3 = abs(vb - vc);
            const int mdiff0 = vcheck == 1 ? min(d0, d1) : vcheck == 2 ? ((d0 + d1 + 1) >> 1) : max(d0, d1);
            const int mdiff1 = vcheck == 1 ? min(d2, d3) : vcheck == 2 ? ((d2 + d3 + 1) >> 1) : max(d2, d3);
            const float a0 = mdiff0 / vthresh0;
            const float a1 = mdiff1 / vthresh1;
            const int dircv = hp ? (abs(dirc) >> 1) : abs(dirc);
            const float a2 = max((vthresh2 - dircv) / vthresh2, 0.0f);
            const float a = min(max(max(a0, a1), a2), 1.0f);
            tline[x] = (int)((1.0 - a)*dstp[x] + a * cint);
          }
          memcpy(dstp, tline, width - MARGIN_H * 2);
        }
        srcp += 2 * spitch;
        dstp += 2 * dpitch;
        if (scpp)
          scpp += 2 * scpitch;
        dstpd += dpitch;
      }
    }
  }
  PVideoFrame dst = env->NewVideoFrame(vi);
  dstPF->copyTo(dst, vi);
  return dst;
}

void eedi3::copyPad(int n, int fn, IScriptEnvironment *env)
{
  const int off = 1 - fn;
  PVideoFrame src = child->GetFrame(n, env);
  if (!dh)
  {
    if (vi.IsYV12())
    {
      const int plane[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
      for (int b = 0; b < 3; ++b)
        env->BitBlt(srcPF->GetPtr(b) + srcPF->GetPitch(b)*(MARGIN_V + off) + MARGIN_H,
          srcPF->GetPitch(b) * 2,
          src->GetReadPtr(plane[b]) + src->GetPitch(plane[b])*off,
          src->GetPitch(plane[b]) * 2, src->GetRowSize(plane[b]),
          src->GetHeight(plane[b]) >> 1);
    }
    else if (vi.IsYUY2())
    {
      srcPF->convYUY2to422(src->GetReadPtr() + src->GetPitch()*off,
        srcPF->GetPtr(0) + srcPF->GetPitch(0)*(MARGIN_V + off) + MARGIN_H,
        srcPF->GetPtr(1) + srcPF->GetPitch(1)*(MARGIN_V + off) + MARGIN_H,
        srcPF->GetPtr(2) + srcPF->GetPitch(2)*(MARGIN_V + off) + MARGIN_H,
        src->GetPitch() * 2, srcPF->GetPitch(0) * 2, srcPF->GetPitch(1) * 2,
        vi.width, vi.height >> 1);
    }
    else
    {
      srcPF->convRGB24to444(src->GetReadPtr() + (vi.height - 1 - off)*src->GetPitch(),
        srcPF->GetPtr(0) + srcPF->GetPitch(0)*(MARGIN_V + off) + MARGIN_H,
        srcPF->GetPtr(1) + srcPF->GetPitch(1)*(MARGIN_V + off) + MARGIN_H,
        srcPF->GetPtr(2) + srcPF->GetPitch(2)*(MARGIN_V + off) + MARGIN_H,
        -src->GetPitch() * 2, srcPF->GetPitch(0) * 2, srcPF->GetPitch(1) * 2,
        vi.width, vi.height >> 1);
    }
  }
  else
  {
    if (vi.IsYV12())
    {
      const int plane[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
      for (int b = 0; b < 3; ++b)
        env->BitBlt(srcPF->GetPtr(b) + srcPF->GetPitch(b)*(MARGIN_V + off) + MARGIN_H,
          srcPF->GetPitch(b) * 2, src->GetReadPtr(plane[b]),
          src->GetPitch(plane[b]), src->GetRowSize(plane[b]),
          src->GetHeight(plane[b]));
    }
    else if (vi.IsYUY2())
    {
      srcPF->convYUY2to422(src->GetReadPtr(),
        srcPF->GetPtr(0) + srcPF->GetPitch(0)*(MARGIN_V + off) + MARGIN_H,
        srcPF->GetPtr(1) + srcPF->GetPitch(1)*(MARGIN_V + off) + MARGIN_H,
        srcPF->GetPtr(2) + srcPF->GetPitch(2)*(MARGIN_V + off) + MARGIN_H,
        src->GetPitch(), srcPF->GetPitch(0) * 2, srcPF->GetPitch(1) * 2,
        vi.width, vi.height >> 1);
    }
    else
    {
      srcPF->convRGB24to444(src->GetReadPtr() + ((vi.height >> 1) - 1)*src->GetPitch(),
        srcPF->GetPtr(0) + srcPF->GetPitch(0)*(MARGIN_V + off) + MARGIN_H,
        srcPF->GetPtr(1) + srcPF->GetPitch(1)*(MARGIN_V + off) + MARGIN_H,
        srcPF->GetPtr(2) + srcPF->GetPitch(2)*(MARGIN_V + off) + MARGIN_H,
        -src->GetPitch(), srcPF->GetPitch(0) * 2, srcPF->GetPitch(1) * 2,
        vi.width, vi.height >> 1);
    }
  }
  for (int b = 0; b < 3; ++b)
  {
    uint8_t *dstp = srcPF->GetPtr(b);
    const int dst_pitch = srcPF->GetPitch(b);
    const int height = srcPF->GetHeight(b);
    const int width = srcPF->GetWidth(b);
    dstp += (MARGIN_V + off)*dst_pitch;
    for (int y = MARGIN_V + off; y < height - MARGIN_V; y += 2)
    {
      for (int x = 0; x < MARGIN_H; ++x)
        dstp[x] = dstp[MARGIN_H * 2 - x];
      int c = 2;
      for (int x = width - MARGIN_H; x < width; ++x, c += 2)
        dstp[x] = dstp[x - c];
      dstp += dst_pitch * 2;
    }
    dstp = srcPF->GetPtr(b);
    for (int y = off; y < MARGIN_V; y += 2)
      env->BitBlt(dstp + y * dst_pitch, dst_pitch,
        dstp + (MARGIN_V * 2 - y)*dst_pitch, dst_pitch, width, 1);
    int c = 2 + 2 * off;
    for (int y = height - MARGIN_V + off; y < height; y += 2, c += 4)
      env->BitBlt(dstp + y * dst_pitch, dst_pitch,
        dstp + (y - c)*dst_pitch, dst_pitch, width, 1);
  }
}

void	eedi3::copyMask(int n, int fn, IScriptEnvironment *env)
{
  const int off = (dh) ? 0 : fn;
  const int mul = (dh) ? 1 : 2;
  PVideoFrame src = mclip->GetFrame(n, env);
  if (vi.IsYV12())
  {
    const int plane[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    for (int b = 0; b < 3; ++b)
      env->BitBlt(
        mcpPF->GetPtr(b),
        mcpPF->GetPitch(b),
        src->GetReadPtr(plane[b]) + src->GetPitch(plane[b]) * off,
        src->GetPitch(plane[b]) * mul,
        src->GetRowSize(plane[b]),
        mcpPF->GetHeight(b)
      );
  }
  else if (vi.IsYUY2())
  {
    mcpPF->convYUY2to422(
      src->GetReadPtr() + src->GetPitch() * off,
      mcpPF->GetPtr(0),
      mcpPF->GetPtr(1),
      mcpPF->GetPtr(2),
      src->GetPitch() * mul,
      mcpPF->GetPitch(0),
      mcpPF->GetPitch(1),
      vi.width,
      mcpPF->GetHeight(0)
    );
  }
  else
  {
    mcpPF->convRGB24to444(
      src->GetReadPtr() + (src->GetHeight() - 1 - off) * src->GetPitch(),
      mcpPF->GetPtr(0),
      mcpPF->GetPtr(1),
      mcpPF->GetPtr(2),
      -src->GetPitch() * mul,
      mcpPF->GetPitch(0),
      mcpPF->GetPitch(1),
      vi.width,
      mcpPF->GetHeight(0)
    );
  }
}

AVSValue __cdecl Create_eedi3(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  if (!args[0].IsClip())
    env->ThrowError("eedi3:  arg 0 must be a clip!");
  VideoInfo vi = args[0].AsClip()->GetVideoInfo();
  if (!vi.IsYV12() && !vi.IsYUY2() && !vi.IsRGB24())
    env->ThrowError("eedi3:  only YV12, YUY2, and RGB24 input are supported!");
  const bool dh = args[2].AsBool(false);
  if ((vi.height & 1) && !dh)
    env->ThrowError("eedi3:  height must be mod 2 when dh=false (%d)!", vi.height);
  return new eedi3(args[0].AsClip(), args[1].AsInt(-1), args[2].AsBool(false),
    args[3].AsBool(true), args[4].AsBool(true), args[5].AsBool(true),
    float(args[6].AsFloat(0.2f)), float(args[7].AsFloat(0.25f)), float(args[8].AsFloat(20.0f)),
    args[9].AsInt(2), args[10].AsInt(20), args[11].AsBool(false), args[12].AsBool(true),
    args[13].AsBool(true), args[14].AsInt(2), float(args[15].AsFloat(32.0f)),
    float(args[16].AsFloat(64.0f)), float(args[17].AsFloat(4.0f)), args[18].IsClip() ?
    args[18].AsClip() : NULL, args[19].AsInt(0), args[20].IsClip() ? args[20].AsClip() : NULL,
    args[21].AsInt(0), env);
}

AVSValue __cdecl Create_eedi3_rpow2(AVSValue args, void* user_data, IScriptEnvironment *env)
{
  if (!args[0].IsClip())
    env->ThrowError("eedi3_rpow2:  arg 0 must be a clip!");
  VideoInfo vi = args[0].AsClip()->GetVideoInfo();
  if (!vi.IsYV12() && !vi.IsYUY2() && !vi.IsRGB24())
    env->ThrowError("eedi3_rpow2:  only YV12, YUY2, and RGB24 input are supported!");
  if (vi.IsYUY2() && (vi.width & 3))
    env->ThrowError("eedi3_rpow2:  for yuy2 input width must be mod 4 (%d)!", vi.width);
  const int rfactor = args[1].AsInt(-1);
  const float alpha = float(args[2].AsFloat(0.2f));
  const float beta = float(args[3].AsFloat(0.25f));
  const float gamma = float(args[4].AsFloat(20.0f));
  const int nrad = args[5].AsInt(2);
  const int mdis = args[6].AsInt(20);
  const bool hp = args[7].AsBool(false);
  const bool ucubic = args[8].AsBool(true);
  const bool cost3 = args[9].AsBool(true);
  const int vcheck = args[10].AsInt(2);
  const float vthresh0 = float(args[11].AsFloat(32.0f));
  const float vthresh1 = float(args[12].AsFloat(64.0f));
  const float vthresh2 = float(args[13].AsFloat(4.0f));
  PClip sclip = NULL;
  const char *cshift = args[14].AsString("");
  const int fwidth = args[15].IsInt() ? args[15].AsInt() : rfactor * vi.width;
  const int fheight = args[16].IsInt() ? args[16].AsInt() : rfactor * vi.height;
  const float ep0 = args[17].IsFloat() ? float(args[17].AsFloat()) : -FLT_MAX;
  const float ep1 = args[18].IsFloat() ? float(args[18].AsFloat()) : -FLT_MAX;
  const int threads = args[19].AsInt(0);
  const int opt = args[20].AsInt(0);
  if (rfactor < 2 || rfactor > 1024)
    env->ThrowError("eedi3_rpow2:  2 <= rfactor <= 1024, and rfactor be a power of 2!\n");
  int rf = 1, ct = 0;
  while (rf < rfactor)
  {
    rf *= 2;
    ++ct;
  }
  if (rf != rfactor)
    env->ThrowError("eedi3_rpow2:  2 <= rfactor <= 1024, and rfactor be a power of 2!\n");
  if (alpha < 0.0f || alpha > 1.0f)
    env->ThrowError("eedi3_rpow2:  0 <= alpha <= 1!\n");
  if (beta < 0.0f || beta > 1.0f)
    env->ThrowError("eedi3_rpow2:  0 <= beta <= 1!\n");
  if (alpha + beta > 1.0f)
    env->ThrowError("eedi3_rpow2:  0 <= alpha+beta <= 1!\n");
  if (gamma < 0.0f)
    env->ThrowError("eedi3_rpow2:  0 <= gamma!\n");
  if (nrad < 0 || nrad > 3)
    env->ThrowError("eedi3_rpow2:  0 <= nrad <= 3!\n");
  if (mdis < 1)
    env->ThrowError("eedi3_rpow2:  1 <= mdis!\n");
  if (vcheck > 0 && (vthresh0 <= 0.0f || vthresh1 <= 0.0f || vthresh2 <= 0.0f))
    env->ThrowError("eedi3_rpow2:  0 < vthresh0 , 0 < vthresh1 , 0 < vthresh2!\n");
  AVSValue v = args[0].AsClip();
  try
  {
    double hshift = 0.0, vshift = 0.0;
    if (vi.IsRGB24())
    {
      for (int i = 0; i < ct; ++i)
      {
        v = new eedi3(v.AsClip(), i == 0 ? 1 : 0, true, true, true, true, alpha,
          beta, gamma, nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0,
          vthresh1, vthresh2, sclip, threads, 0, opt, env);
        v = env->Invoke("TurnRight", v).AsClip();
        v = new eedi3(v.AsClip(), i == 0 ? 1 : 0, true, true, true, true, alpha,
          beta, gamma, nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0,
          vthresh1, vthresh2, sclip, threads, 0, opt, env);
        v = env->Invoke("TurnLeft", v).AsClip();
      }
      hshift = vshift = -0.5;
    }
    else if (vi.IsYV12())
    {
      for (int i = 0; i < ct; ++i)
      {
        v = new eedi3(v.AsClip(), i == 0 ? 1 : 0, true, true, true, true, alpha, beta,
          gamma, nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1,
          vthresh2, sclip, threads, 0, opt, env);
        v = env->Invoke("TurnRight", v).AsClip();
        // always use field=1 to keep chroma/luma horizontal alignment
        v = new eedi3(v.AsClip(), 1, true, true, true, true, alpha, beta, gamma,
          nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2,
          sclip, threads, 0, opt, env);
        v = env->Invoke("TurnLeft", v).AsClip();
      }
      // Correct chroma shift (it's always 1/2 pixel upwards).
      // Need a cache here because v/vc will both request from this point.
      v = env->Invoke("InternalCache", v).AsClip();
      v.AsClip()->SetCacheHints(CACHE_GET_RANGE, 2);
      AVSValue sargs[7] = { v, vi.width*rfactor, vi.height*rfactor, 0.0, -0.5,
        vi.width*rfactor, vi.height*rfactor };
      const char *nargs[7] = { 0, 0, 0, "src_left", "src_top",
        "src_width", "src_height" };
      AVSValue vc = env->Invoke("Spline36Resize", AVSValue(sargs, 7), nargs).AsClip();
      AVSValue margs[2] = { v, vc };
      v = env->Invoke("MergeChroma", AVSValue(margs, 2)).AsClip();
      for (int i = 0; i < ct; ++i)
        hshift = hshift * 2.0 - 0.5;
      vshift = -0.5;
    }
    else
    {
      // Unfortunately, turnleft()/turnright() can't preserve YUY2 chroma, so we convert
      // U/V planes to Y planes in separate clips and process them that way.
      AVSValue vu = env->Invoke("UtoY", v).AsClip();
      AVSValue vv = env->Invoke("VtoY", v).AsClip();
      for (int i = 0; i < ct; ++i)
      {
        v = new eedi3(v.AsClip(), i == 0 ? 1 : 0, true, true, false, false, alpha, beta, gamma,
          nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2, sclip, threads, 0, opt, env);
        v = env->Invoke("TurnRight", v).AsClip();
        // always use field=1 to keep chroma/luma horizontal alignment
        v = new eedi3(v.AsClip(), 1, true, true, false, false, alpha, beta, gamma, nrad,
          mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2, sclip, threads, 0, opt, env);
        v = env->Invoke("TurnLeft", v).AsClip();
      }
      for (int i = 0; i < ct; ++i)
      {
        vu = new eedi3(vu.AsClip(), i == 0 ? 1 : 0, true, true, false, false, alpha, beta,
          gamma, nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1,
          vthresh2, sclip, threads, 0, opt, env);
        vu = env->Invoke("TurnRight", vu).AsClip();
        // always use field=1 to keep chroma/luma horizontal alignment
        vu = new eedi3(vu.AsClip(), 1, true, true, false, false, alpha, beta, gamma,
          nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2,
          sclip, threads, 0, opt, env);
        vu = env->Invoke("TurnLeft", vu).AsClip();
      }
      for (int i = 0; i < ct; ++i)
      {
        vv = new eedi3(vv.AsClip(), i == 0 ? 1 : 0, true, true, false, false, alpha, beta,
          gamma, nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2,
          sclip, threads, 0, opt, env);
        vv = env->Invoke("TurnRight", vv).AsClip();
        // always use field=1 to keep chroma/luma horizontal alignment
        vv = new eedi3(vv.AsClip(), 1, true, true, false, false, alpha, beta, gamma,
          nrad, mdis, hp, ucubic, cost3, vcheck, vthresh0, vthresh1, vthresh2,
          sclip, threads, 0, opt, env);
        vv = env->Invoke("TurnLeft", vv).AsClip();
      }
      AVSValue ytouvargs[3] = { vu, vv, v };
      v = env->Invoke("YtoUV", AVSValue(ytouvargs, 3)).AsClip();
      for (int i = 0; i < ct; ++i)
        hshift = hshift * 2.0 - 0.5;
      vshift = -0.5;
    }
    if (cshift[0])
    {
      int type = 0;
      if (_strnicmp(cshift, "blackmanresize", 14) == 0 ||
        _strnicmp(cshift, "lanczosresize", 13) == 0 ||
        _strnicmp(cshift, "sincresize", 10) == 0)
        type = 1;
      else if (_strnicmp(cshift, "gaussresize", 11) == 0)
        type = 2;
      else if (_strnicmp(cshift, "bicubicresize", 13) == 0)
        type = 3;
      if (!type || (type != 3 && ep0 == -FLT_MAX) ||
        (type == 3 && ep0 == -FLT_MAX && ep1 == -FLT_MAX))
      {
        AVSValue sargs[7] = { v, fwidth, fheight, hshift, vshift,
          vi.width*rfactor, vi.height*rfactor };
        const char *nargs[7] = { 0, 0, 0, "src_left", "src_top",
          "src_width", "src_height" };
        v = env->Invoke(cshift, AVSValue(sargs, 7), nargs).AsClip();
      }
      else if (type != 3 || min(ep0, ep1) == -FLT_MAX)
      {
        AVSValue sargs[8] = { v, fwidth, fheight, hshift, vshift,
          vi.width*rfactor, vi.height*rfactor, type == 1 ? AVSValue((int)(ep0 + 0.5f)) :
          (type == 2 ? ep0 : max(ep0, ep1)) };
        const char *nargs[8] = { 0, 0, 0, "src_left", "src_top",
          "src_width", "src_height", type == 1 ? "taps" : (type == 2 ? "p" : (max(ep0, ep1) == ep0 ? "b" : "c")) };
        v = env->Invoke(cshift, AVSValue(sargs, 8), nargs).AsClip();
      }
      else
      {
        AVSValue sargs[9] = { v, fwidth, fheight, hshift, vshift,
          vi.width*rfactor, vi.height*rfactor, ep0, ep1 };
        const char *nargs[9] = { 0, 0, 0, "src_left", "src_top",
          "src_width", "src_height", "b", "c" };
        v = env->Invoke(cshift, AVSValue(sargs, 9), nargs).AsClip();
      }
    }
  }
  catch (IScriptEnvironment::NotFound)
  {
    env->ThrowError("eedi3_rpow2:  error using env->invoke (function not found)!\n");
  }
  return v;
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;

  env->AddFunction("eedi3", "c[field]i[dh]b[Y]b[U]b[V]b[alpha]f[beta]f[gamma]f[nrad]i[mdis]i" \
    "[hp]b[ucubic]b[cost3]b[vcheck]i[vthresh0]f[vthresh1]f[vthresh2]f[sclip]c[threads]i[mclip]c[opt]i",
    Create_eedi3, 0);
  env->AddFunction("eedi3_rpow2", "c[rfactor]i[alpha]f[beta]f[gamma]f[nrad]i[mdis]i[hp]b" \
    "[ucubic]b[cost3]b[vcheck]i[vthresh0]f[vthresh1]f[vthresh2]f[cshift]s[fwidth]i" \
    "[fheight]i[ep0]f[ep1]f[threads]i[opt]i",
    Create_eedi3_rpow2, 0);
  return "eedi3 plugin";
}
