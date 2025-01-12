#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

// Function Prototypes for Fluid Solver

// Density step
void dens_step_cuda(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt);

// Velocity step
void vel_step_cuda(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt);

// Add source
void add_source_cuda(int M, int N, int O, float *d_x, float *d_s, float dt);

// Diffusion
void diffuse_cuda(int M, int N, int O, int b, float *d_x, float *d_x0, float diff, float dt);

// Advection
void advect_cuda(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt);

// Projection
void project_cuda(int M, int N, int O, float *u, float *v, float *w, float *p, float *div);

// Boundary conditions
void set_bnd_cuda(int M, int N, int O, int b, float *x);

#endif // FLUID_SOLVER_H
