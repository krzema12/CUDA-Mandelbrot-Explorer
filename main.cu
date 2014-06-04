#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <complex>
#include <ctime>
#include "Common.h"
#include "Timer.h"
#include "WindowInit.h"
#include "kernel.cuh"
using namespace std;

// http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__DEVICE_g028e5b0474379eaf5f5d54657d48600b.html#g028e5b0474379eaf5f5d54657d48600b

#define CPU	0

GtkWidget *window;
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

float centerX, centerY, scale;
int iterations = 512;

int bufferWidth = 640;
int bufferHeight = 480;

int lastCanvasWidth = 0;
int lastCanvasHeight = 0;

void updateStatusBar(double time);

void setDefaultView()
{
	centerX = -0.7f;
	centerY = 0.0f;
	scale = 3.0f;
}

static gboolean draw_cb(GtkWidget *widget, cairo_t *cr, gpointer data)
{   
	gdk_cairo_set_source_pixbuf(cr, pixbuf, 0, 0);
	cairo_paint(cr);
	cairo_fill(cr);

	return FALSE;
}

void updateBuffer()
{
	int bufferPos = 0;
	
	Timer timer;
	timer.start();

	double ratio = (double)bufferWidth/(double)bufferHeight;
	
	if (currentDevice == CPU)
	{
		for (int y=0; y<bufferHeight; y++)
		{
			for (int x=0; x<bufferWidth; x++)
			{		
				complex<double> c((double)(x - bufferWidth/2)*scale*ratio/(double)(bufferWidth - 1) + centerX,
					(double)(y - bufferHeight/2)*scale/(double)(bufferHeight - 1) + centerY);
				complex<double> z(0.0, 0.0);
				int i = 510;
			
				// checking if we're in the cardioid
				//double q = (real(c) - 0.25)*(real(c) - 0.25) + imag(c)*imag(c);
			
				//if ((q*(q + (real(c) - 0.25)) >= 0.25*imag(c)*imag(c)) && ((real(c) + 1)*(real(c) + 1) + imag(c)*imag(c) >= 0.0625))
				{
					for (i=0; i<iterations; i++)
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
		dim3 threads(blockWidth, blockHeight);
		dim3 grid((bufferWidth + (blockWidth - 1))/blockWidth, (bufferHeight + (blockHeight - 1))/blockHeight);

		mandelbrotPixel<<<grid, threads>>>(deviceBuffer, devicePalette, bufferWidth, bufferHeight, centerX, centerY, scale, iterations);
		
		//cudaError_t err = cudaSuccess; 
		//err = cudaGetLastError();
		//cerr << "Failed to launch kernel (error code %s)! " << cudaGetErrorString(err);
		
		cudaMemcpy(rawBuffer, deviceBuffer, bufferWidth*bufferHeight*3, cudaMemcpyDeviceToHost);
	}
	
	timer.stop();
	gtk_widget_queue_draw(da);
	updateStatusBar(timer.getElapsedTimeInSec());
}

void updateStatusBar(double time)
{
	ostringstream newStatus;
	newStatus << fixed << setprecision(5) << "Center: " << centerX << " " << showpos << centerY << "i   Scale: "
		<< noshowpos << scale << "   Iterations: " << iterations << "   Viewport: " << bufferWidth << "x" << bufferHeight << "   Time: " << time << " ms   |   ";
	
	if (currentDevice == 0)
		newStatus << "CPU";
	else
	{
		int memInMB = (deviceProps[currentDevice - 1].totalGlobalMem + 1024*1024 - 1)/(1024*1024);
		newStatus << deviceProps[currentDevice - 1].name << "    " << memInMB << " MB    CUDA Compute Capability: " <<
			deviceProps[currentDevice - 1].major << "." << deviceProps[currentDevice - 1].minor;
	}
		
	gtk_statusbar_push(GTK_STATUSBAR(statusBar), 0, newStatus.str().c_str());
}

gboolean canvasFrameChanged(GtkWindow *window, GdkEvent *event, gpointer data)
{
	if (lastCanvasWidth != event->configure.width ||
		lastCanvasHeight != event->configure.height)
	{
		// reallocating local frame buffer
		delete[] rawBuffer;
		
		if (pixbuf != NULL)
			g_object_unref(pixbuf);
		
		bufferWidth = event->configure.width;
		bufferHeight = event->configure.height;
		
		rawBuffer = new byte[bufferWidth*bufferHeight*3];
		pixbuf = gdk_pixbuf_new_from_data(rawBuffer, GDK_COLORSPACE_RGB,
			FALSE, 8, bufferWidth, bufferHeight, bufferWidth*3, NULL, NULL);

		// (re)allocating CUDA device's frame buffer

		if (currentDevice != CPU)
		{
			if (deviceBuffer != 0)
				cudaFree(deviceBuffer);
			
			cudaMalloc((void**)&deviceBuffer, bufferWidth*bufferHeight*3);
		}

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

			// frame buffer
			if (deviceBuffer != 0)
				cudaFree(deviceBuffer);
			
			cudaMalloc((void**)&deviceBuffer, bufferWidth*bufferHeight*3);

			// palette buffer
			if (devicePalette != NULL)
				cudaFree(devicePalette);

			cudaMalloc((void**)&devicePalette, (iterations + 2)*3);
			cudaMemcpy(devicePalette, currentPalette, (iterations + 2)*3, cudaMemcpyHostToDevice);
		}

		updateBuffer();
	}
}

void create_dialog(GtkWindow *window, char *title, char *message)
{
    GtkWidget *dialog, *label, *content_area;

    /* New label for dialog content */
    label = gtk_label_new(message);

    /* Make a new dialog with an 'OK' button */
    dialog = gtk_dialog_new_with_buttons(title, window, GTK_DIALOG_DESTROY_WITH_PARENT, GTK_STOCK_OK, GTK_RESPONSE_NONE, NULL);

    /* Add label to dialog */
    content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_container_add(GTK_CONTAINER(content_area), label);

    /* Destroy dialog properly */
    g_signal_connect(dialog, "response", G_CALLBACK(gtk_widget_destroy), dialog);

    /* Set dialog to not resize. */
    gtk_window_set_resizable(GTK_WINDOW(dialog), FALSE);

    gtk_widget_show_all(dialog);
}

void openHelp(GtkWidget *widget, int whichWindow)
{
	if (whichWindow == 0)
		create_dialog((GtkWindow*)window, "Usage", "Arrow keys: moving the view up/down and left/right\nPlus\\minus keys: zooming in\\out\n\nQ\\A: increasing\\decreasing the number of iterations\nW\\S: increasing\\decreasing the number of iterations by 100");
	else
		create_dialog((GtkWindow*)window, "About", "Mandelbrot Explorer\nby Piotr Krzeminski, 131546\n\nThis application has been created as a project\nfor \"CUDA\\CELL processing\" university course.");
}

int amplify(int val)
{
	float floatVal = (float)val/255.0f;
	float amplified = sqrtf(floatVal);
	
	return (int)(amplified*255.0f);
}

void generatePalette(int paletteID)
{
	if (currentPalette != NULL)
		delete[] currentPalette;

	currentPalette = new byte[(iterations + 2)*3];

	int arrayPos = 0;

	// grayscale
	if (paletteID == 0)
	{
		for (int i=0; i<=iterations; i++)
		{
			currentPalette[arrayPos++] = 255 - amplify(i*255/iterations);
			currentPalette[arrayPos++] = 255 - amplify(i*255/iterations);
			currentPalette[arrayPos++] = 255 - amplify(i*255/iterations);
		}
	}
	else if (paletteID == 1)
	{
		for (int i=0; i<iterations; i++)
		{
			currentPalette[arrayPos++] = amplify(i <= iterations/2 ? 0 : (i - iterations/2)*2*255/iterations);
			currentPalette[arrayPos++] = amplify(i <= iterations/2 ? i*2*255/iterations : 255);
			currentPalette[arrayPos++] = amplify(i <= iterations/2 ? 0 : (i - iterations/2)*2*255/iterations);
		}
			
		currentPalette[arrayPos++] = 0;
		currentPalette[arrayPos++] = 0;
		currentPalette[arrayPos++] = 0;			
	}

	if (currentDevice != CPU)
	{
		if (devicePalette != NULL)
			cudaFree(devicePalette);

		cudaMalloc((void**)&devicePalette, (iterations + 2)*3);
		cudaMemcpy(devicePalette, currentPalette, (iterations + 2)*3, cudaMemcpyHostToDevice);
	}

	currentPaletteID = paletteID;
}

void paletteChanged(GtkWidget *widget, int paletteID)
{
	if (paletteID != currentPaletteID)
	{
		generatePalette(paletteID);
		updateBuffer();
	}
}

gboolean onKeyPress(GtkWidget *widget, GdkEventKey *event, gpointer user_data)
{
	switch (event->keyval)
	{
		case GDK_KEY_Left:
			centerX -= 0.05f*scale;
			break;
		case GDK_KEY_Right:
			centerX += 0.05f*scale;
			break;
		case GDK_KEY_Up:
			centerY -= 0.05f*scale;
			break;
		case GDK_KEY_Down:
			centerY += 0.05f*scale;
			break;
		case GDK_KEY_equal:
		case GDK_KEY_KP_Add:
			scale /= 1.1f;
			break;
		case GDK_KEY_minus:
		case GDK_KEY_KP_Subtract:
			scale *= 1.1f;
			break;
		case GDK_KEY_r:
			setDefaultView();
			break;
		case GDK_KEY_q:
			iterations++;
			generatePalette(currentPaletteID);
			break;
		case GDK_KEY_a:
			iterations = max(1, iterations - 1);
			generatePalette(currentPaletteID);
			break;
		case GDK_KEY_w:
			iterations += 100;
			generatePalette(currentPaletteID);
			break;
		case GDK_KEY_s:
			iterations = max(1, iterations - 100);
			generatePalette(currentPaletteID);
			break;
		default:
			return FALSE;
	}
	
	updateBuffer();
	gtk_widget_queue_draw(da);
	
	return FALSE;
}

void initWindow()
{
	// creating a window
	window = createWindow(G_CALLBACK(onKeyPress));
	
	// creating a main menu
	GtkWidget *menuBar = gtk_menu_bar_new();

	// adding subsequent submenus
	gtk_menu_shell_append(GTK_MENU_SHELL(menuBar), createDeviceMenu(G_CALLBACK(menuitem_response)));
	gtk_menu_shell_append(GTK_MENU_SHELL(menuBar), createPaletteMenu(G_CALLBACK(paletteChanged)));
	gtk_menu_shell_append(GTK_MENU_SHELL(menuBar), createHelpMenu(G_CALLBACK(openHelp)));
	
	// creating a drawing area
	da = createDrawingArea(G_CALLBACK(draw_cb), G_CALLBACK(canvasFrameChanged));
	
	// creating a status bar
	statusBar = gtk_statusbar_new();

	// adding all the elements to the window, stacked vertically
	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	gtk_container_add(GTK_CONTAINER(window), vbox);
	gtk_box_pack_start(GTK_BOX(vbox), menuBar, FALSE, FALSE, 0);
	gtk_box_pack_start(GTK_BOX(vbox), da, TRUE, TRUE, 0);
	gtk_box_pack_start(GTK_BOX(vbox), statusBar, FALSE, FALSE, 3);
	
	// displaying the window
	gtk_widget_show_all(window);
}

int main(int argc, char *argv[])
{
	gtk_init(&argc, &argv);

	// setting initial values
	setDefaultView();
	generatePalette(1);
	
	initWindow();

	// the main loop
	gtk_main();

	return 0;
}
