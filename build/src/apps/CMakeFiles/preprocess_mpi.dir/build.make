# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zluo/work/relion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zluo/work/relion/build

# Include any dependencies generated for this target.
include src/apps/CMakeFiles/preprocess_mpi.dir/depend.make

# Include the progress variables for this target.
include src/apps/CMakeFiles/preprocess_mpi.dir/progress.make

# Include the compile flags for this target's objects.
include src/apps/CMakeFiles/preprocess_mpi.dir/flags.make

src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o: src/apps/CMakeFiles/preprocess_mpi.dir/flags.make
src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o: ../src/apps/preprocess_mpi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zluo/work/relion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o"
	cd /home/zluo/work/relion/build/src/apps && /data/ni_data/EMAN2/bin/mpicxx   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o -c /home/zluo/work/relion/src/apps/preprocess_mpi.cpp

src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.i"
	cd /home/zluo/work/relion/build/src/apps && /data/ni_data/EMAN2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zluo/work/relion/src/apps/preprocess_mpi.cpp > CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.i

src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.s"
	cd /home/zluo/work/relion/build/src/apps && /data/ni_data/EMAN2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zluo/work/relion/src/apps/preprocess_mpi.cpp -o CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.s

src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o.requires:

.PHONY : src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o.requires

src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o.provides: src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o.requires
	$(MAKE) -f src/apps/CMakeFiles/preprocess_mpi.dir/build.make src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o.provides.build
.PHONY : src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o.provides

src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o.provides.build: src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o


# Object files for target preprocess_mpi
preprocess_mpi_OBJECTS = \
"CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o"

# External object files for target preprocess_mpi
preprocess_mpi_EXTERNAL_OBJECTS =

bin/relion_preprocess_mpi: src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o
bin/relion_preprocess_mpi: src/apps/CMakeFiles/preprocess_mpi.dir/build.make
bin/relion_preprocess_mpi: lib/librelion_lib.so
bin/relion_preprocess_mpi: /data/ni_data/EMAN2/lib/libfftw3.so
bin/relion_preprocess_mpi: /usr/local/cuda/lib64/libcufft.so
bin/relion_preprocess_mpi: /data/ni_data/EMAN2/lib/libmpi_cxx.so
bin/relion_preprocess_mpi: /data/ni_data/EMAN2/lib/libmpi.so
bin/relion_preprocess_mpi: lib/librelion_gpu_util.so
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/libfltk_images.a
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/libfltk_forms.a
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/libfltk.a
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/libSM.so
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/libICE.so
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/libX11.so
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/libXext.so
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/libm.so
bin/relion_preprocess_mpi: /usr/local/cuda/lib64/libcudart_static.a
bin/relion_preprocess_mpi: /usr/lib/x86_64-linux-gnu/librt.so
bin/relion_preprocess_mpi: src/apps/CMakeFiles/preprocess_mpi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zluo/work/relion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/relion_preprocess_mpi"
	cd /home/zluo/work/relion/build/src/apps && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/preprocess_mpi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/apps/CMakeFiles/preprocess_mpi.dir/build: bin/relion_preprocess_mpi

.PHONY : src/apps/CMakeFiles/preprocess_mpi.dir/build

src/apps/CMakeFiles/preprocess_mpi.dir/requires: src/apps/CMakeFiles/preprocess_mpi.dir/preprocess_mpi.cpp.o.requires

.PHONY : src/apps/CMakeFiles/preprocess_mpi.dir/requires

src/apps/CMakeFiles/preprocess_mpi.dir/clean:
	cd /home/zluo/work/relion/build/src/apps && $(CMAKE_COMMAND) -P CMakeFiles/preprocess_mpi.dir/cmake_clean.cmake
.PHONY : src/apps/CMakeFiles/preprocess_mpi.dir/clean

src/apps/CMakeFiles/preprocess_mpi.dir/depend:
	cd /home/zluo/work/relion/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zluo/work/relion /home/zluo/work/relion/src/apps /home/zluo/work/relion/build /home/zluo/work/relion/build/src/apps /home/zluo/work/relion/build/src/apps/CMakeFiles/preprocess_mpi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/apps/CMakeFiles/preprocess_mpi.dir/depend

