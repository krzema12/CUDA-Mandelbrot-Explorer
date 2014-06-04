#include "kernel.cuh"

//#define BITMASK_OPTIMALIZATION

__global__ void mandelbrotPixel(byte *output, byte *palette, int width, int height, float centerX, float centerY, float scale, int iterations)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
    
	if ((x >= width) || (y >= height))
		return;

	float ratio = (float)width/(float)height;
    	
	float cReal, cImag;
	cReal = (float)(x - width/2)*scale*ratio/(float)(width - 1) + centerX;
	cImag = (float)(y - height/2)*scale/(float)(height - 1) + centerY;
    
	float zReal = 0.0f, zImag = 0.0, z2Real, z2Imag;
	
	int i;

#ifdef BITMASK_OPTIMALIZATION
	int iters = 0;
	int doneMask = 0;
	int itersIsEmptyMask = 0;
#endif
	
#pragma unroll
	for (i = 0; i<iterations; i++)
	{
		z2Real = zReal*zReal - zImag*zImag + cReal;
		z2Imag = 2.0f*zReal*zImag + cImag;
		
		zReal = z2Real;
		zImag = z2Imag;

#ifdef BITMASK_OPTIMALIZATION
		doneMask = ~((zReal*zReal + zImag*zImag > 4.0f) - 1);	// time to exit the loop  =>  0xFFFFFFFF
		itersIsEmptyMask = (iters != 0) - 1;	// iters == 0  =>  0xFFFFFFFF
		iters = (doneMask & itersIsEmptyMask & i) | ((~itersIsEmptyMask) & iters);
#else
		if (zReal*zReal + zImag*zImag > 4.0f)
			break;
#endif
	}

#ifdef BITMASK_OPTIMALIZATION
	int paletteIndex = iters*3;
#else
	int paletteIndex = i*3;
#endif

	int bufferPos = (width*y + x)*3;

	output[bufferPos++] = palette[paletteIndex++];
	output[bufferPos++] = palette[paletteIndex++];
	output[bufferPos++] = palette[paletteIndex++];
}
