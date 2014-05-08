#ifndef WINDOWINIT_H
#define WINDOWINIT_H

#include <gtk/gtk.h>
#include <cuda_runtime.h>

#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 600

GtkWidget* createWindow(GCallback onKeyPressCallback);
GtkWidget* createDeviceMenu(GCallback menuitem_response);
GtkWidget* createPaletteMenu(GCallback paletteChanged);
GtkWidget* createHelpMenu(GCallback openHelp);
GtkWidget* createDrawingArea(GCallback redrawCallback, GCallback canvasFrameChanged);

#endif