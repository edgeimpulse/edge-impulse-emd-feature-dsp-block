# If you're not using the Armadillo wrapper, replace `armadillo` with linker commands
# for the BLAS and LAPACK libraries that you are using.

TARGET := main
SRC := main.cpp eemd.cpp
LIBS_NAME := armadillo

CXX := g++
CXXFLAGS += -std=c++11 -Wall -Wextra -O3 -DNDEBUG -fopenmp
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
CXXFLAGS += -std=c++11 -Wall -Wextra -g -O0
GSL_FLAGS := $(shell pkg-config --libs --cflags gsl) -DHAVE_INLINE
LDFLAGS  += -fopenmp
LDFLAGS += -L. # if installed locally.
# Add header directories for any includes that aren't on the
# default compiler search path.
INCLFLAGS := -I .
CXXFLAGS += $(INCLFLAGS)

OBJS := $(SRC:.cpp=.o)
LIBS := $(addprefix -l,$(LIBS_NAME))
CLEAN_LIST := $(TARGET) $(OBJS)

# default rule
default: all

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS) $(GSL_FLAGS) $(LIBS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)
