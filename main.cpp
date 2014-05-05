/* COMPILE USING:  gcc -Wextra -o cairo1 `pkg-config --cflags --libs gtk+-3.0` cairo1.c */
#include <gtk/gtk.h>

#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 768

static gboolean draw_cb(GtkWidget *widget, cairo_t *cr, gpointer data)
{   
   /* Set color for background */
   cairo_set_source_rgb(cr, 1, 1, 1);
   /* fill in the background color*/
   cairo_paint(cr);
     
   /* set color for rectangle */
   cairo_set_source_rgb(cr, 0.42, 0.65, 0.80);
   /* set the line width */
   cairo_set_line_width(cr,6);
   /* draw the rectangle's path beginning at 3,3 */
   cairo_rectangle (cr, 3, 3, 100, 100);
   /* stroke the rectangle's path with the chosen color so it's actually visible */
   cairo_stroke(cr);

   /* draw circle */
   cairo_set_source_rgb(cr, 0.17, 0.63, 0.12);
   cairo_set_line_width(cr,2);
   cairo_arc(cr, 150, 210, 20, 0, 2*G_PI);
   cairo_stroke(cr);

   /* draw horizontal line */
   cairo_set_source_rgb(cr, 0.77, 0.16, 0.13);
   cairo_set_line_width(cr, 6);
   cairo_move_to(cr, 80,160);
   cairo_line_to(cr, 200, 160);
   cairo_stroke(cr);

   return FALSE;
}

int main(int argc, char *argv[])
{
   GtkWidget *window;
   GtkWidget *da;

   gtk_init(&argc, &argv);

   window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
   gtk_window_set_title((GtkWindow*)window, "Mandelbrot Explorer by Piotr Krzemiński, 131546");
   g_signal_connect(window, "destroy", G_CALLBACK (gtk_main_quit), NULL);

   da = gtk_drawing_area_new();
   gtk_widget_set_size_request(da, WINDOW_WIDTH, WINDOW_HEIGHT);
   g_signal_connect(da, "draw", G_CALLBACK(draw_cb),  NULL);

   gtk_container_add(GTK_CONTAINER (window), da);
   gtk_widget_show(da);
   gtk_widget_show(window);

   gtk_main();

   return 0;
}