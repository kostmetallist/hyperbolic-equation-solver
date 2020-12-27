# Hyperbolic equation solver

A parallel program for calculating numerical solution of
[PDE](https://en.wikipedia.org/wiki/Partial_differential_equation).
Uses MPI and OpenMP to unleash the power of multiprocessing and multithreading.

## Problem

Full task explanation is provided in the
[assignment document](https://github.com/kostmetallist/hyperbolic-equation-solver/blob/master/doc/task.pdf).
Briefly, it is required to design and implement a programmatic solution for given differential equation using
modern parallel programming technologies.

## Compilation

This project uses `Makefile`. There are several targets inside the instructions file. Most likely you will
need only `gnu` and `clean` targets as the rest relate to specific computing complexes with proprietary IBM
compilers. To prepare a binary, be sure to have OpenMP installed in your system and run `make gnu`.

## Execution

On Linux systems, there is a `mpirun` utility dedicated for running MPI-driven programs. Use the following
format:

```
mpirun -np <processes_number> <binary_name> [<binary_arguments>]
```

## Troubleshooting

In case of any suggestions, bug reports or questions, please to refer to the
[Issues](https://github.com/kostmetallist/hyperbolic-equation-solver/issues) section.
