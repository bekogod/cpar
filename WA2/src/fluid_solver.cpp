#include "fluid_solver.h"
#include <cmath>

#define IX(i, j, k) ((k) + (M + 2) * (j) + (M + 2) * (N + 2) * (i))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
  x[i] += dt * s[i];
  }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  #pragma omp parallel
  {

    #pragma omp for collapse(2) schedule(static)
    for (i = 1; i <= M; i++) {
      for (j = 1; j <= N; j++) {
        x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
      }
    }

    #pragma omp for collapse(2) schedule(static)
    for (i = 1; i <= N; i++) {
      for (j = 1; j <= O; j++) {
        x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
        x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
      }
    }

    #pragma omp for collapse(2) schedule(static)
    for (i = 1; i <= M; i++) {
      for (j = 1; j <= O; j++) {
        x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
        x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
      }
    }

    #pragma omp single //SINGLE É UM BOCADO REDUNDANTE AQUI, MAS É PARA GARANTIR QUE O CÓDIGO SEJA EXECUTADO APENAS UMA VEZ
    {
      x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
      x[IX(M + 1, 0, 0)] =
          0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
      x[IX(0, N + 1, 0)] =
          0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
      x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                        x[IX(M + 1, N + 1, 1)]);
    }
  }
}


// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7; 
    float max_c;      
    int l = 0;       

    do {
        max_c = 0.0f;

        #pragma omp parallel for reduction(max:max_c) schedule(static) collapse(2)
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                for (int k = 1 + (i + j) % 2; k <= O; k += 2) {
                    float old_x = x[IX(i, j, k)];
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
                    float change = fabs(x[IX(i, j, k)] - old_x);
                    if (change > max_c) max_c = change;
                }
            }
        }

        #pragma omp parallel for reduction(max:max_c) schedule(static) collapse(2)
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                for (int k = 1 + (i + j + 1) % 2; k <= O; k += 2) {
                    float old_x = x[IX(i, j, k)];
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
                    float change = fabs(x[IX(i, j, k)] - old_x);
                    if (change > max_c) max_c = change;
                }
            }
        }

        set_bnd(M, N, O, b, x);
    } while (max_c > tol && ++l < 20);
}



// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

        // Clamp to grid boundaries
        x = MAX(0.5f, MIN(M + 0.5f, x));
        y = MAX(0.5f, MIN(N + 0.5f, y));
        z = MAX(0.5f, MIN(O + 0.5f, z));

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[IX(i, j, k)] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
      }
    }
  }
  set_bnd(M, N, O, b, d);
}


// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {

  #pragma omp parallel for collapse(3) schedule(static)
  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        div[IX(i, j, k)] =
            -0.5f * (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
                     v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
                     w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
            MAX(M, MAX(N, O));
        p[IX(i, j, k)] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  #pragma omp parallel for collapse(3) schedule(static)
  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
      }
    }
  }

  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}


// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}