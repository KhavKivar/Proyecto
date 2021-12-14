
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>

#define pi 3.14159265359

void load_image(char *fname, int Nx, int Ny, float *img)
{
  FILE *fp;

  fp = fopen(fname, "r");

  for (int i = 0; i < Ny; i++)
  {
    for (int j = 0; j < Nx; j++)
      fscanf(fp, "%f ", &img[i * Nx + j]);
    fscanf(fp, "\n");
  }

  fclose(fp);
}

void save_image(char *fname, int Nx, int Ny, float *img)
{
  FILE *fp;

  fp = fopen(fname, "w");
  
  for (int i = 0; i < Ny; i++)
  {
    for (int j = 0; j < Nx; j++){
      fprintf(fp, "%10.3f ", img[i * Nx + j]);
    
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}



void calculate_kernel(int kernel_size, float sigma, float *kernel)
{

  int Nk2 = kernel_size * kernel_size;
  float x, y, center;

  center = (kernel_size - 1) / 2.0;

  for (int i = 0; i < Nk2; i++)
  {
    x = (float)(i % kernel_size) - center;
    y = (float)(i / kernel_size) - center;
    kernel[i] = -(1.0 / pi * pow(sigma, 4)) * (1.0 - 0.5 * (x * x + y * y) / (sigma * sigma)) * exp(-0.5 * (x * x + y * y) / (sigma * sigma));
  }
}


void conv_img_cpu(float *img, float *kernel, float *imgf, int Nx, int Ny, int kernel_size)
{

  float sum = 0;
  int center = (kernel_size - 1) / 2;
  ;
  int ii, jj;

  for (int i = center; i < (Ny - center); i++)
    for (int j = center; j < (Nx - center); j++)
    {
      sum = 0;
      for (int ki = 0; ki < kernel_size; ki++)
        for (int kj = 0; kj < kernel_size; kj++)
        {
          ii = kj + j - center;
          jj = ki + i - center;
          sum += img[jj * Nx + ii] * kernel[ki * kernel_size + kj];
        }
      

      imgf[i * Nx + j] = sum;
    }
}


__global__ void conv_img_gpu(float *img, float *kernel, float *imgf, int Nx, int Ny, int kernel_size)
{

  int tid = threadIdx.x;
  int iy = blockIdx.x + (kernel_size - 1) / 2;

  int idx = iy * Nx;
  int K2 = kernel_size * kernel_size;
  int center = (kernel_size - 1) / 2;
  int ii, jj;
  float sum = 0.0;
  extern __shared__ float sdata[];

  if (idx < Ny*Nx )
  {
    for(int i = center;i<(Nx - center);i++){
      sum = 0;
      for (int ki = 0; ki < kernel_size; ki++)
        for (int kj = 0; kj < kernel_size; kj++)
        {
          ii = kj + i - center;
          jj = ki + iy - center;
          if(jj * Nx + ii < Nx*Ny)
          sum += img[jj * Nx + ii] * kernel[ki * kernel_size + kj];
        }

      imgf[idx+i] = sum;
   }
  }
}




#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}







int main(int argc, char *argv[])
{
  cudaEvent_t start, stop;
  clock_t t1, t2;
  double ms;


  float milliseconds = 0;
  int Nx, Ny;
  int kernel_size;
  float sigma;
  char finput[256], foutput[256], foutput_cpu[256];
  int Nblocks, Nthreads;

  sprintf(finput, "dog.dat");

  sprintf(foutput, "gpu_output.dat");
  sprintf(foutput_cpu, "cpu_output.dat");


  Nx = 750;
  Ny = 750;

  kernel_size = 3;
  sigma = 0.55;
  


  float *img, *imgf, *imgf_cpu, *kernel;

  img = (float *)malloc(Nx * Ny * sizeof(float));
  imgf = (float *)malloc(Nx * Ny * sizeof(float));
  imgf_cpu = (float *)malloc(Nx * Ny * sizeof(float));
  kernel = (float *)malloc(kernel_size * kernel_size * sizeof(float));

  //Cargamos la imagen y calculamos los valores del kennel segun el kernel_size
  load_image(finput, Nx, Ny, img);
  calculate_kernel(kernel_size, sigma, kernel);

  t1 = clock();
 



  conv_img_cpu(img, kernel, imgf_cpu, Nx, Ny, kernel_size);

  t2 = clock();
  ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
  printf("Tiempo (CPU): %f[ms]\n", ms);



 



  float *d_img, *d_imgf, *d_kernel;

  cudaMalloc(&d_img, Nx * Ny * sizeof(float));
  cudaMalloc(&d_imgf, Nx * Ny * sizeof(float));
  cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));




  cudaMemcpy(d_img, img, Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  
  int block_size = 256;


  
  int grid_size = (Ny - (kernel_size - 1));


  conv_img_gpu<<<grid_size, 1, kernel_size * kernel_size * sizeof(float)>>>(d_img, d_kernel, d_imgf, Nx, Ny, kernel_size);    


  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  
  printf("Tiempo (GPU): %f ms\n", milliseconds);

  gpuErrchk(cudaMemcpy(imgf, d_imgf, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost));




  save_image(foutput, Nx, Ny, imgf);
  save_image(foutput_cpu, Nx, Ny, imgf_cpu);



  free(img);
  free(imgf);
  free(kernel);

  cudaFree(d_img);
  cudaFree(d_imgf);
  cudaFree(d_kernel);
}
