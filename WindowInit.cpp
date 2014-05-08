#include "WindowInit.h"

extern int devicesCount;
extern cudaDeviceProp *deviceProps;

GtkWidget* createWindow(GCallback onKeyPressCallback)
{
	GtkWidget *newWindow = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_default_size((GtkWindow*)newWindow, WINDOW_WIDTH, WINDOW_HEIGHT);
	gtk_window_set_title((GtkWindow*)newWindow, "Mandelbrot Explorer");
	g_signal_connect(newWindow, "destroy", G_CALLBACK(gtk_main_quit), NULL);
	g_signal_connect(newWindow, "key_press_event", onKeyPressCallback, NULL);

	return newWindow;
}

GtkWidget* createDeviceMenu(GCallback menuitem_response)
{
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

	return deviceMenuItem;
}

GtkWidget* createPaletteMenu(GCallback paletteChanged)
{
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

	return paletteMenuItem;
}

GtkWidget* createHelpMenu(GCallback openHelp)
{
	GtkWidget *helpMenu = gtk_menu_new();
	GtkWidget *helpMenuItem = gtk_menu_item_new_with_label("Help");
	GtkWidget *usageMenuItem = gtk_menu_item_new_with_label("Usage");
	GtkWidget *aboutMenuItem = gtk_menu_item_new_with_label("About");
	
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(helpMenuItem), helpMenu);
	gtk_menu_shell_append(GTK_MENU_SHELL(helpMenu), usageMenuItem);
	g_signal_connect(usageMenuItem, "activate", G_CALLBACK(openHelp), (gpointer)0);
	gtk_menu_shell_append(GTK_MENU_SHELL(helpMenu), aboutMenuItem);
	g_signal_connect(aboutMenuItem, "activate", G_CALLBACK(openHelp), (gpointer)1);

	return helpMenuItem;
}

GtkWidget* createDrawingArea(GCallback redrawCallback, GCallback canvasFrameChanged)
{
	GtkWidget* drawingArea = gtk_drawing_area_new();
	g_signal_connect(drawingArea, "draw", redrawCallback, NULL);
	g_signal_connect(drawingArea, "configure-event", canvasFrameChanged, NULL);

	return drawingArea;
}
