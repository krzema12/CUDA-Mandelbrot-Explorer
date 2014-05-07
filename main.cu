#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <complex>
#include <ctime>
#include <cuda_runtime.h>
#include "Timer.h"
using namespace std;
typedef unsigned char byte;

// http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__DEVICE_g028e5b0474379eaf5f5d54657d48600b.html#g028e5b0474379eaf5f5d54657d48600b

#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 768

#define CPU	0

GtkWidget *da;
GtkWidget *statusBar;
GdkPixbuf *pixbuf;

byte *rawBuffer;
byte *deviceBuffer;

byte *currentPalette;
byte *devicePalette;

int currentPaletteID = 0;

int devicesCount, currentDevice = 0;
cudaDeviceProp *deviceProps;

double centerX, centerY, scale;

int bufferWidth = 640;
int bufferHeight = 480;

int lastCanvasWidth = 0;
int lastCanvasHeight = 0;

void updateStatusBar(double time);

void setDefaultView()
{
	centerX = -0.7;
	centerY = 0.0;
	scale = 3.0;
}

static gboolean draw_cb(GtkWidget *widget, cairo_t *cr, gpointer data)
{   
	gdk_cairo_set_source_pixbuf(cr, pixbuf, 0, 0);
	cairo_paint(cr);
	cairo_fill(cr);

	return FALSE;
}

__global__ void mandelbrotPixel(byte *output, byte *palette, int width, int height, double centerX, double centerY, double scale)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if ((x >= width) || (y >= height))
    	return;
    	
    double cReal, cImag;
    cReal = (double)(x - width/2)*scale/(double)(width - 1) + centerX;
    cImag = (double)(y - height/2)*scale/(double)(height - 1) + centerY;
    
	double zReal = 0.0f, zImag = 0.0f, z2Real, z2Imag;
	
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
		
	output[bufferPos++] = palette[i*3];
	output[bufferPos++] = palette[i*3 + 1];
	output[bufferPos++] = palette[i*3 + 2];
}

void updateBuffer()
{
	int bufferPos = 0;
	
	Timer timer;
	timer.start();
	
	if (currentDevice == CPU)
	{
		for (int y=0; y<bufferHeight; y++)
		{
			for (int x=0; x<bufferWidth; x++)
			{		
				complex<double> c((double)(x - bufferWidth/2)*scale/(double)(bufferWidth - 1) + centerX,
					(double)(y - bufferHeight/2)*scale/(double)(bufferHeight - 1) + centerY);
				complex<double> z(0.0, 0.0);
				int i = 510;
			
				// checking if we're in the cardioid
				double q = (real(c) - 0.25)*(real(c) - 0.25) + imag(c)*imag(c);
			
				if ((q*(q + (real(c) - 0.25)) >= 0.25*imag(c)*imag(c)) && ((real(c) + 1)*(real(c) + 1) + imag(c)*imag(c) >= 0.0625))
				{
					for (i=0; i<510; i++)
					{
						z = z*z + c;
					
						if (real(z)*real(z) + imag(z)*imag(z) > 4.0)
							break;
					}
				}
	
				rawBuffer[bufferPos++] = currentPalette[i*3];
				rawBuffer[bufferPos++] = currentPalette[i*3 + 1];
				rawBuffer[bufferPos++] = currentPalette[i*3 + 2];
			}
		}
	}
	else
	{
		if (deviceBuffer != 0)
			cudaFree(deviceBuffer);
			
		cudaMalloc((void**)&deviceBuffer, bufferWidth*bufferHeight*3);
		cudaMalloc((void**)&devicePalette, 512*3);
		
		cudaMemcpy(devicePalette, currentPalette, 512*3, cudaMemcpyHostToDevice);
		
		dim3 threads(8, 8);
		dim3 grid((bufferWidth + 7)/8, (bufferHeight + 7)/8);
	
		mandelbrotPixel<<<grid, threads>>>(deviceBuffer, devicePalette, bufferWidth, bufferHeight, centerX, centerY, scale);
		
		//cudaError_t err = cudaSuccess; 
		//err = cudaGetLastError();
		//cerr << "Failed to launch kernel (error code %s)! " << cudaGetErrorString(err);
		
		cudaMemcpy(rawBuffer, deviceBuffer, bufferWidth*bufferHeight*3, cudaMemcpyDeviceToHost);
	}
	
	timer.stop();
	updateStatusBar(timer.getElapsedTimeInSec());
}

void updateStatusBar(double time)
{
	ostringstream newStatus;
	newStatus << fixed << setprecision(5) << "Center: " << centerX << " " << showpos << centerY << "i   Scale: "
		<< noshowpos << scale << "   Time: " << time << " ms   |   ";
	
	if (currentDevice == 0)
		newStatus << "CPU";
	else
	{
		int memInMB = (deviceProps[currentDevice - 1].totalGlobalMem + 1024*1024 - 1)/(1024*1024);
		newStatus << deviceProps[currentDevice - 1].name << "    " << memInMB << " MB";
	}
		
	gtk_statusbar_push(GTK_STATUSBAR(statusBar), 0, newStatus.str().c_str());
	gtk_widget_queue_draw(statusBar);
}

gboolean canvasFrameChanged(GtkWindow *window, GdkEvent *event, gpointer data)
{
	if (lastCanvasWidth != event->configure.width ||
		lastCanvasHeight != event->configure.height)
	{
		delete[] rawBuffer;
		
		if (pixbuf != NULL)
			g_object_unref(pixbuf);
		
		bufferWidth = event->configure.width;
		bufferHeight = event->configure.height;
		
		rawBuffer = new byte[bufferWidth*bufferHeight*3];
		pixbuf = gdk_pixbuf_new_from_data(rawBuffer, GDK_COLORSPACE_RGB,
			FALSE, 8, bufferWidth, bufferHeight, bufferWidth*3, NULL, NULL);			

		updateBuffer();
	
		lastCanvasWidth = event->configure.width;
		lastCanvasHeight = event->configure.height;
	}
	
	return FALSE;
}

void menuitem_response(GtkWidget *widget, int device)
{
	if (device != currentDevice)
	{
		// changing the current device
		currentDevice = device;
	
		if (device != CPU)
		{
			cudaSetDevice(device - 1);
		}

		updateBuffer();
	}
}

int amplify(int val)
{
	float floatVal = (float)val/255.0f;
	float amplified = sqrtf(floatVal);
	
	return (int)(amplified*255.0f);
}

void paletteChanged(GtkWidget *widget, int paletteID)
{
	if (paletteID != currentPaletteID)
	{
		int arrayPos = 0;
		
		// grayscale
		if (paletteID == 0)
		{
			for (int i=0; i<511; i++)
			{
				currentPalette[arrayPos++] = 255 - amplify(i/2);
				currentPalette[arrayPos++] = 255 - amplify(i/2);
				currentPalette[arrayPos++] = 255 - amplify(i/2);
			}
		}
		else if (paletteID == 1)
		{
			for (int i=0; i<510; i++)
			{
				currentPalette[arrayPos++] = amplify(i <= 255 ? 0 : i - 256);
				currentPalette[arrayPos++] = amplify(i <= 255 ? i : 255);
				currentPalette[arrayPos++] = amplify(i <= 255 ? 0 : i - 256);
			}
			
			currentPalette[arrayPos++] = 0;
			currentPalette[arrayPos++] = 0;
			currentPalette[arrayPos++] = 0;			
		}
		
		currentPaletteID = paletteID;
		updateBuffer();
	}
}

gboolean onKeyPress(GtkWidget *widget, GdkEventKey *event, gpointer user_data)
{
	switch (event->keyval)
	{
		case GDK_KEY_Left:
			centerX -= 0.05*scale;
			break;
		case GDK_KEY_Right:
			centerX += 0.05*scale;
			break;
		case GDK_KEY_Up:
			centerY -= 0.05*scale;
			break;
		case GDK_KEY_Down:
			centerY += 0.05*scale;
			break;
		case GDK_KEY_equal:
		case GDK_KEY_KP_Add:
			scale /= 1.1;
			break;
		case GDK_KEY_minus:
		case GDK_KEY_KP_Subtract:
			scale *= 1.1;
			break;
		case GDK_KEY_r:
			setDefaultView();
			break;
		default:
			return FALSE;
	}
	
	updateBuffer();

	gtk_widget_queue_draw(da);
	
	return FALSE;
}

int main(int argc, char *argv[])
{
	setDefaultView();
	
	// ----------------

	currentPalette = new byte[512*3];
	int arrayPos = 0;
	
	for (int i=0; i<510; i++)
	{
		currentPalette[arrayPos++] = amplify(i <= 255 ? 0 : i - 256);
		currentPalette[arrayPos++] = amplify(i <= 255 ? i : 255);
		currentPalette[arrayPos++] = amplify(i <= 255 ? 0 : i - 256);
	}
	
	currentPalette[arrayPos++] = 0;
	currentPalette[arrayPos++] = 0;
	currentPalette[arrayPos++] = 0;		
	
	// ----------------

	gtk_init(&argc, &argv);
	
	// creating a window
	GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_default_size((GtkWindow*)window, WINDOW_WIDTH, WINDOW_HEIGHT);
	gtk_window_set_title((GtkWindow*)window, "Mandelbrot Explorer");
	g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
	g_signal_connect(window, "key_press_event", G_CALLBACK(onKeyPress), NULL);
	
	// creating the main menu
	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	gtk_container_add(GTK_CONTAINER(window), vbox);
	
	GtkWidget *menuBar = gtk_menu_bar_new();
	GtkWidget *deviceMenu = gtk_menu_new();
	
	// "Device" menu and radio buttons
	GtkWidget *deviceMenuItem = gtk_menu_item_new_with_label("Device");
	GSList *devicesRadioGroup = NULL;
	
	GtkWidget *cpuMenuItem = gtk_radio_menu_item_new_with_label(devicesRadioGroup, "CPU");
	gtk_menu_shell_append(GTK_MENU_SHELL(deviceMenu), cpuMenuItem);
	g_signal_connect(cpuMenuItem, "activate", G_CALLBACK(menuitem_response), (gpointer)0);
	devicesRadioGroup = gtk_radio_menu_item_get_group(GTK_RADIO_MENU_ITEM(cpuMenuItem));
	gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(cpuMenuItem), TRUE);
	
	// getting info about installed CUDA-capable devices
	
	cudaGetDeviceCount(&devicesCount);
	deviceProps = new cudaDeviceProp[devicesCount];
	
	for (int i=0; i<devicesCount; i++)
	{
		cudaGetDeviceProperties(&deviceProps[i], i);

		GtkWidget *deviceMenuItem = gtk_radio_menu_item_new_with_label(devicesRadioGroup, deviceProps[i].name);
		gtk_menu_shell_append(GTK_MENU_SHELL(deviceMenu), deviceMenuItem);
		g_signal_connect(deviceMenuItem, "activate", G_CALLBACK(menuitem_response), (gpointer)(i + 1));
				
		devicesRadioGroup = gtk_radio_menu_item_get_group(GTK_RADIO_MENU_ITEM(deviceMenuItem));
	}
	
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(deviceMenuItem), deviceMenu);
	gtk_menu_shell_append(GTK_MENU_SHELL(menuBar), deviceMenuItem);
	
	// "Palette" menu and radio buttons
	GtkWidget *paletteMenu = gtk_menu_new();
	GtkWidget *paletteMenuItem = gtk_menu_item_new_with_label("Palette");
	GSList *palettesRadioGroup = NULL;
	GtkWidget *grayscaleMenuItem = gtk_radio_menu_item_new_with_label(palettesRadioGroup, "Grayscale");
	
	palettesRadioGroup = gtk_radio_menu_item_get_group(GTK_RADIO_MENU_ITEM(grayscaleMenuItem));
	GtkWidget *blackGreenWhiteMenuItem = gtk_radio_menu_item_new_with_label(palettesRadioGroup, "Black-green-white");
	
	// set "Black-green-white" as currently selected
	gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(blackGreenWhiteMenuItem), TRUE);
	
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(paletteMenuItem), paletteMenu);
	gtk_menu_shell_append(GTK_MENU_SHELL(paletteMenu), grayscaleMenuItem);
	g_signal_connect(grayscaleMenuItem, "activate", G_CALLBACK(paletteChanged), (gpointer)0);
	gtk_menu_shell_append(GTK_MENU_SHELL(paletteMenu), blackGreenWhiteMenuItem);
	g_signal_connect(blackGreenWhiteMenuItem, "activate", G_CALLBACK(paletteChanged), (gpointer)1);
	gtk_menu_shell_append(GTK_MENU_SHELL(menuBar), paletteMenuItem);
	
	// "Help" menu
	GtkWidget *helpMenu = gtk_menu_new();
	GtkWidget *helpMenuItem = gtk_menu_item_new_with_label("Help");
	GtkWidget *usageMenuItem = gtk_menu_item_new_with_label("Usage");
	GtkWidget *aboutMenuItem = gtk_menu_item_new_with_label("About");
	
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(helpMenuItem), helpMenu);
	gtk_menu_shell_append(GTK_MENU_SHELL(helpMenu), usageMenuItem);
	gtk_menu_shell_append(GTK_MENU_SHELL(helpMenu), aboutMenuItem);
	gtk_menu_shell_append(GTK_MENU_SHELL(menuBar), helpMenuItem);
	
	gtk_box_pack_start(GTK_BOX(vbox), menuBar, FALSE, FALSE, 0);
	
	// creating a drawing area
	da = gtk_drawing_area_new();
	g_signal_connect(da, "draw", G_CALLBACK(draw_cb), NULL);
	g_signal_connect(da, "configure-event", G_CALLBACK(canvasFrameChanged), NULL);
	
	gtk_box_pack_start(GTK_BOX(vbox), da, TRUE, TRUE, 0);

	// creating a status bar
	statusBar = gtk_statusbar_new();
	gtk_box_pack_start(GTK_BOX(vbox), statusBar, FALSE, FALSE, 3);
	
	gtk_widget_show_all(window);

	// the main loop
	gtk_main();

	return 0;
}
