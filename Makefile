TARGET1 = svm
OUTPUT1 = bin/test_$(TARGET1)

TARGET2 = nu_svm
OUTPUT2 = bin/test_$(TARGET2)

CC = g++

CFLAGS = -std=c++11 -c -ggdb -Wall -O3 -fopenmp -fpermissive -Wno-reorder -Wno-comment -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS = -fopenmp

BASE_DIR := /usr/local

DEFS =	-DDLIB_JPEG_SUPPORT \
	-DDLIB_PNG_SUPPORT

INCLUDE_DIRS = \
	`pkg-config --cflags opencv` \
	`pkg-config --cflags eigen3` \
	-I./src \
	-I./lib/dlib-18.14/

LIBS =  \
	`pkg-config --libs opencv` \
	`pkg-config --libs eigen3` \
	-ljpeg -lpng -lX11 -lpthread
#	-L./lib/dlib-18.14/build/ -ldlib

OBJECTS_COMMON = src/datahandler.o src/dataconverter.o \
	src/svmtestsuite.o src/utils.o
OBJECTS1 = src/main.o $(OBJECTS_COMMON)
OBJECTS2 = src/test_fhog.o

DEPS1 = $(OBJECTS1:%.o=%.P)
DEPS2 = $(OBJECTS2:%.o=%.P)

.PHONY: all execute clean

all: $(TARGET1)

$(TARGET1): $(OBJECTS1)
	$(CC) -o $(OUTPUT1) $(LDFLAGS) $(OBJECTS1) $(LIB_DIRS) $(LIBS)

$(TARGET2): $(OBJECTS2)
	$(CC) -o $(OUTPUT2) $(LDFLAGS) $(OBJECTS2) $(LIB_DIRS) $(LIBS)

%.o : %.cpp
	$(CC) $(CFLAGS) $(DEFS) $(INCLUDE_DIRS) -MD $< -o $@
	@cp $*.d $*.P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
		rm -f $*.d

-include $(OBJECTS1:%.o=%.P)
-include $(OBJECTS2:%.o=%.P)

clean:
	rm -f $(OBJECTS1) $(DEPS1) $(OUTPUT1) $(OBJECTS2) $(DEPS2) $(OUTPUT2)

execute:
	./$(TARGET1)
