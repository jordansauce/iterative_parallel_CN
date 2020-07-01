# Iterative Parallel Crank-Nicolson method

We demonstrate the rapid convergence of an iterative approximation that enables the Crank-Nicolson method for solving partial differential equations to be parallelised with arbitrarily small error, so long as the temporal step size is kept sufficiently small compared to the square of the spatial step size. Unlike the original Crank-Nicolson method, our method can handle non-linear terms. We apply the new method specifically to linear and non-linear 1-D Schrodinger equations, but it can be applied to any parabolic partial differential equation, such as those used to describe heat flow, diffusion, or dynamics of financial markets. The extension of the method to handle cyclic boundary conditions or 3-D wavefunctions is straightforward.

Included is the [iterative_parallel_CN.cpp] code file, which can be compiled with 

    $ g++ -fopenmp iterative_parallel_CN.cpp -std=c++11 -lm -Wall -O3 -o iterative_parallel_CN.exe
    
The input file [iterative_parallel_CN.inp] contains the variables to be given to the program when run:

    $ ./iterative_parallel_CN.exe
    
The results are written to multiple raw files [iterative_parallel_CN_Xiterations_XXXXXX.dat], and one summary file [iterative_parallel_CN_summary.dat]. The raw files can be turned into an animation with gnuplot using the file animate.gpi:

    $ gnuplot animate.gpi
    
This will produce a GIF of whatever scenario you put into the .inp file (by default, a gaussian oscillating in a harmonic trap).


Keywords:parallel, implicit, iterative, partial differential equations, Schr ̈odinger equation, Schrodinger, non-linear, TDSE

