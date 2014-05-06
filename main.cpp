#include <gtk/gtk.h>
#include <gdk/gdkkeysyms.h>
#include <sstream>
#include <iomanip>
using namespace std;

#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 768

GtkWidget *da;
GtkWidget *statusBar;

double x = 0.0;
double y = 0.0;

static gboolean draw_cb(GtkWidget *widget, cairo_t *cr, gpointer data)
{   
   // Set color for background
   cairo_set_source_rgb(cr, 1, 1, 1);
   // fill in the background color
   cairo_paint(cr);
   
   cairo_translate(cr, x, y);
     
   // set color for rectangle
   cairo_set_source_rgb(cr, 0.42, 0.65, 0.80);
   // set the line width
   cairo_set_line_width(cr,6);
   // draw the rectangle's path beginning at 3,3
   cairo_rectangle (cr, 3, 3, 100, 100);
   // stroke the rectangle's path with the chosen color so it's actually visible
   cairo_stroke(cr);

   // draw circle
   cairo_set_source_rgb(cr, 0.17, 0.63, 0.12);
   cairo_set_line_width(cr,2);
   cairo_arc(cr, 150, 210, 20, 0, 2*G_PI);
   cairo_stroke(cr);

   // draw horizontal line
   cairo_set_source_rgb(cr, 0.77, 0.16, 0.13);
   cairo_set_line_width(cr, 6);
   cairo_move_to(cr, 80,160);
   cairo_line_to(cr, 200, 160);
   cairo_stroke(cr);

   return FALSE;
}

gboolean onKeyPress(GtkWidget *widget, GdkEventKey *event, gpointer user_data)
{
	switch (event->keyval)
	{
		case GDK_KEY_Left:
			x -= 5.0;
			break;
		case GDK_KEY_Right:
			x += 5.0;
			break;
		case GDK_KEY_Up:
			y -= 5.0;
			break;
		case GDK_KEY_Down:
			y += 5.0;
			break;			
		default:
			return FALSE;
	}
	
	ostringstream newStatus;
	newStatus << fixed << setprecision(2) << "Center: " << x << " " << showpos << y << "i   Scale: 1.00";
	
	gtk_statusbar_push(GTK_STATUSBAR(statusBar), 0, newStatus.str().c_str());	
	gtk_widget_queue_draw(da);
	
	return FALSE;
}

int main(int argc, char *argv[])
{
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
	GtkWidget *dummyDevice1MenuItem = gtk_radio_menu_item_new_with_label(devicesRadioGroup, "Dummy device 1");
	devicesRadioGroup = gtk_radio_menu_item_get_group(GTK_RADIO_MENU_ITEM(dummyDevice1MenuItem));
	GtkWidget *dummyDevice2MenuItem = gtk_radio_menu_item_new_with_label(devicesRadioGroup, "Dummy device 2");
	// set "dummyDevice2MenuItem" to currently selected
	gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(dummyDevice2MenuItem), TRUE);
	
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(deviceMenuItem), deviceMenu);
	gtk_menu_shell_append(GTK_MENU_SHELL(deviceMenu), dummyDevice1MenuItem);
	gtk_menu_shell_append(GTK_MENU_SHELL(deviceMenu), dummyDevice2MenuItem);
	gtk_menu_shell_append(GTK_MENU_SHELL(menuBar), deviceMenuItem);
	
	// "Palette" menu and radio buttons
	GtkWidget *paletteMenu = gtk_menu_new();
	GtkWidget *paletteMenuItem = gtk_menu_item_new_with_label("Palette");
	GSList *palettesRadioGroup = NULL;
	GtkWidget *palette1MenuItem = gtk_radio_menu_item_new_with_label(palettesRadioGroup, "Grayscale");
	palettesRadioGroup = gtk_radio_menu_item_get_group(GTK_RADIO_MENU_ITEM(palette1MenuItem));
	GtkWidget *palette2MenuItem = gtk_radio_menu_item_new_with_label(palettesRadioGroup, "Black to green");
	// set "palette1MenuItem" to currently selected
	gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(palette1MenuItem), TRUE);
	
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(paletteMenuItem), paletteMenu);
	gtk_menu_shell_append(GTK_MENU_SHELL(paletteMenu), palette1MenuItem);
	gtk_menu_shell_append(GTK_MENU_SHELL(paletteMenu), palette2MenuItem);
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
	
	gtk_box_pack_start(GTK_BOX(vbox), da, TRUE, TRUE, 0);
	
	// creating a status bar
	statusBar = gtk_statusbar_new();
	gtk_statusbar_push(GTK_STATUSBAR(statusBar), 0, "Center: 0.00 + 0.00i   Scale: 1.00");
	gtk_box_pack_start(GTK_BOX(vbox), statusBar, FALSE, FALSE, 3);
	
	gtk_widget_show_all(window);

	// the main loop
	gtk_main();

	return 0;
}