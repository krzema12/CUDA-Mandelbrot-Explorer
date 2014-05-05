ifeq ($(OS),Windows_NT)
# Windows

main.exe: main.cpp
	gcc -o main.exe main.cpp -mms-bitfields -IC:/gtk3/include/gtk-3.0 -IC:/gtk3/include/cairo -IC:/gtk3/include/pango-1.0 -IC:/gtk3/include/atk-1.0 -IC:/gtk3/include/cairo -IC:/gtk3/include/pixman-1 -IC:/gtk3/include -IC:/gtk3/include/freetype2 -IC:/gtk3/include -IC:/gtk3/include/libpng15 -IC:/gtk3/include/gdk-pixbuf-2.0 -IC:/gtk3/include/libpng15 -IC:/gtk3/include/glib-2.0 -IC:/gtk3/lib/glib-2.0/include -LC:/gtk3/lib -lgtk-3 -lgdk-3 -lgdi32 -limm32 -lshell32 -lole32 -Wl,-luuid -lpangocairo-1.0 -lpangoft2-1.0 -lfreetype -lfontconfig -lpangowin32-1.0 -lgdi32 -lpango-1.0 -lm -latk-1.0 -lcairo-gobject -lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lgobject-2.0 -lglib-2.0 -lintl
	
else
# Linux 
cross_os: cross_os.o    
	gcc -o cross_os cross_os.o `pkg-config --cflags --libs gtk+-2.0` -export-dynamic 
	
cross_os.o: cross_os.asm
	nasm -f elf cross_os.asm 
	
endif
