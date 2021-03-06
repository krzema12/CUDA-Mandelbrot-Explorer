#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "Common.h"

__global__ void mandelbrotPixel(byte *output, byte *palette, int width, int height, float centerX, float centerY, float scale, int iterations);
__global__ void mandelbrotThread(byte *output, byte *palette, int width, int height, int threads, float centerX, float centerY, float scale, int iterations);

#endif