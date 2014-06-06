#ifndef WINDOWINIT_H
#define WINDOWINIT_H

#include <gtk/gtk.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 768

GtkWidget* createWindow(GCallback onKeyPressCallback);
GtkWidget* createDeviceMenu(GCallback menuitem_response);
GtkWidget* createPaletteMenu(GCallback paletteChanged);
GtkWidget* createAntialiasingMenu(GCallback antialiasingChanged);
GtkWidget* createBlockSizeMenu(GCallback blockSizeChanged);
GtkWidget* createHelpMenu(GCallback openHelp);
GtkWidget* createDrawingArea(GCallback redrawCallback, GCallback canvasFrameChanged);

#endif