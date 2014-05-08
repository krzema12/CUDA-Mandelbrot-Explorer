#include "kernel.cuh"

__global__ void mandelbrotPixel(byte *output, byte *palette, int width, int height, float centerX, float centerY, float scale)
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
	
	for (i = 0; i<510; i++)
	{
		z2Real = zReal*zReal - zImag*zImag + cReal;
		z2Imag = 2.0f*zReal*zImag + cImag;
		
		zReal = z2Real;
		zImag = z2Imag;
		
		if (zReal*zReal + zImag*zImag > 4.0f)
			break;
	}
		
	int bufferPos = (width*y + x)*3;
		
	int invert2 = -(x == width/2 || y == height/2);
	int invert = 1 - ((x == width/2 || y == height/2)<<1); // if NO then 1, if YES then -1

	output[bufferPos++] = (0xFF&invert2) + (invert*palette[i*3]);
	output[bufferPos++] = (0xFF&invert2) + (invert*palette[i*3 + 1]);
	output[bufferPos++] = (0xFF&invert2) + (invert*palette[i*3 + 2]);

	//output[bufferPos++] = palette[i*3];
	//output[bufferPos++] = palette[i*3 + 1];
	//output[bufferPos++] = palette[i*3 + 2];
}
