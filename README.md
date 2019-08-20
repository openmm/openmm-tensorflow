OpenMM Neural Network Plugin
============================

This is a plugin for [OpenMM](http://openmm.org) that allows neural networks
to be used for defining forces.  It is implemented with [TensorFlow](https://www.tensorflow.org/).
To use it, you create a TensorFlow graph that takes particle positions as input
and produces forces and energy as output.  This plugin uses the graph to apply
forces to particles during a simulation.

Installation
============

At present this plugin must be compiled from source.  It uses CMake as its build
system.  Before compiling you must install the TensorFlow C API by following the
instructions at https://www.tensorflow.org/install/install_c.  You can then
follow these steps.

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.

5. Set TENSORFLOW_DIR to point to the directory where you installed the TensorFlow C API.

6. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

7. If you plan to build the OpenCL platform, make sure that OPENCL_INCLUDE_DIR and
OPENCL_LIBRARY are set correctly, and that NN_BUILD_OPENCL_LIB is selected.

8. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that NN_BUILD_CUDA_LIB is selected.

9. Press "Configure" again if necessary, then press "Generate".

10. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install` to install the plugin, and `make PythonInstall` to
install the Python wrapper.

Usage
=====

The first step is to create a TensorFlow graph defining the calculation to
perform.  It should have an input called `positions` which will be set to the
particle positions.  It should produce two outputs: one called `forces`
containing the forces to apply to the particles, and one called `energy` with
the potential energy.  This graph must then be saved to a binary protocol
buffer file.  Here is an example of Python code that does this for a very
simple calculation (a harmonic force attracting every particle to the origin).

```python
import tensorflow as tf
graph = tf.Graph()
with graph.as_default():
    positions = tf.placeholder(tf.float32, [None, 3], 'positions')
    energy = tf.reduce_sum(positions**2, name='energy')
    forces = tf.identity(tf.gradients(-energy, positions), name='forces')
tf.train.write_graph(graph, '.', 'graph.pb', as_text=False)
```

The data types of the graph's input and output tensors may be either `float32`
or `float64`.

To use the graph in a simulation, create a `NeuralNetworkForce` object and add
it to your `System`.  The constructor takes the path to the saved graph as an
argument.  For example,

```python
from openmmnn import *
f = NeuralNetworkForce('graph.pb')
system.addForce(f)
```

Alternatively, in Python (but not C++) you can directly pass a `tf.Graph` to the
constructor without writing it to a file first:

```python
f = NeuralNetworkForce(graph)
```

If the graph includes any variables, pass a `tf.Session` as the second argument.
It will use TensorFlow's `freeze_graph` utility to create a frozen version of the
graph in which variables have been replaced with values taken from the session:

```python
f = NeuralNetworkForce(graph, session)
```

When defining the graph to perform a calculation, you may want to apply
periodic boundary conditions.  To do this, call `setUsesPeriodicBoundaryConditions(True)`
on the `NeuralNetworkForce`.  The graph is then expected to contain an input
called `boxvectors` which will contain the current periodic box vectors.  You
can make use of them in whatever way you want for computing the force.  For
example, the following code applies periodic boundary conditions to each
particle position to translate all of them into a single periodic cell.

```python
positions = tf.placeholder(tf.float32, [None, 3], 'positions')
boxvectors = tf.placeholder(tf.float32, [3, 3], 'boxvectors')
boxsize = tf.diag_part(boxvectors)
periodicPositions = positions - tf.floor(positions/boxsize)*boxsize
```

Note that this code assumes a rectangular box.  Applying periodic boundary
conditions with a triclinic box requires a slightly more complicated
calculation.

License
=======

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2018 Stanford University and the Authors.

Authors: Peter Eastman

Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

