
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>
void Read(float ****RImg, int *Count,
          int *M, int *N, const char *filename)
{
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d %d\n", Count, M, N);

    int imsize = (*M) * (*N);

    float ***RImg1 = new float **[*Count];
    for (int j = 0; j < *Count; j++)
    {
        RImg1[j] = new float *[3];
        for (int i = 0; i < 3; i++)
        {
            RImg1[j][i] = new float[imsize];
        }
    }
    for (int k = 0; k < *Count; k++)
    {
        for (int i = 0; i < imsize; i++)
            fscanf(fp, "%f ", &(RImg1[k][0][i]));
        for (int i = 0; i < imsize; i++)
            fscanf(fp, "%f ", &(RImg1[k][1][i]));
        for (int i = 0; i < imsize; i++)
            fscanf(fp, "%f ", &(RImg1[k][2][i]));
    }
    fclose(fp);
    *RImg = RImg1;
}
void Write(float **image,
           int M, int N, const char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", M, N);
    for (int i = 0; i < M * N - 1; i++)
        fprintf(fp, "%f ", image[0][i]);
    fprintf(fp, "%f\n", image[0][M * N - 1]);
    for (int i = 0; i < M * N - 1; i++)
        fprintf(fp, "%f ", image[1][i]);
    fprintf(fp, "%f\n", image[1][M * N - 1]);
    for (int i = 0; i < M * N - 1; i++)
        fprintf(fp, "%f ", image[2][i]);
    fprintf(fp, "%f\n", image[2][M * N - 1]);
    fclose(fp);
}
void funcionCPU(float ***host_images, float **host_images_out, int Count, int M, int N)
{

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < M * N; j++)
            host_images_out[i][j] = 0;
    for (int i = 0; i < Count; i++)

    {

        for (int j = 0; j < M * N; j++)

        {
            host_images_out[0][j] += host_images[i][0][j] / Count;
            host_images_out[1][j] += host_images[i][1][j] / Count;
            host_images_out[2][j] += host_images[i][2][j] / Count;
        }
    }
   }

__global__ void kernel(int Count, int M, int N, float *device_images_R, float *device_images_G, float *device_images_B, float *device_images_R_out, float *device_images_G_out, float *device_images_B_out)
{
   
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < M * N)
    {
        
        float suma1 = 0;
        float suma2 = 0;
        float suma3 = 0;
        int y = (tid % (M * N)) / N;
        int x = tid % N;
        for (int i = 0; i < Count; i++)
        {
            suma1 += device_images_R[x + y * N + i * M * N];
            suma2 += device_images_G[x + y * N + i * M * N];
            suma3 += device_images_B[x + y * N + i * M * N];
        }
        device_images_R_out[tid] = suma1 * (1.0 / Count);
        device_images_G_out[tid] = suma2 * (1.0 / Count);
        device_images_B_out[tid] = suma3 * (1.0 / Count);
    }
}

int main()
{
    
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    int Count, M, N, Mout, Nout;
    float ***host_images;

    float *host_images_R, *host_images_G, *host_images_B;

    float *device_images_R, *device_images_G, *device_images_B;
    float *device_images_R_out, *device_images_G_out, *device_images_B_out;
    float **host_image_out;

    char names[6][3][100] = {
        {"images6.txt\0", "images6CPU.txt\0", "images6GPU.txt\0"},
        {"images5.txt\0", "images5CPU.txt\0", "images5GPU.txt\0"},
        {"images4.txt\0", "images4CPU.txt\0", "images4GPU.txt\0"},
        {"images3.txt\0", "images3CPU.txt\0", "images3GPU.txt\0"},
        {"images2.txt\0", "images2CPU.txt\0", "images2GPU.txt\0"},
        {"images1.txt\0", "images1CPU.txt\0", "images1GPU.txt\0"},
    };

    for (int j = 0; j < 6; j++)
    {
        Read(&host_images, &Count, &M, &N, names[j][0]);
        Mout = M;
        Nout = N;

        host_image_out = new float *[3];
        for (int i = 0; i < 3; i++)
        {
            host_image_out[i] = new float[M * N];
        }

        auto inicio = high_resolution_clock::now();
        funcionCPU(host_images, host_image_out, Count, M, N);
        auto final = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(final - inicio);
        duration<double, std::milli> ms_double = final - inicio;
        std::cout << "Imagen " << 6 - j << std::endl;
        
        std::cout <<"Tiempo CPU: " << ms_double.count() << "[ms]" << std::endl;

        Write(host_image_out, Mout, Nout, names[j][1]);
        for (int i = 0; i < 3; i++)
        {
            delete[] host_image_out[i];
        }
        delete[] host_image_out;

        /*
	      Parte GPU
	     */
        int grid_size, block_size = 256;
        grid_size = (int)ceil((float)Mout * Nout / block_size);

        host_images_R = new float[Count * M * N];
        host_images_G = new float[Count * M * N];
        host_images_B = new float[Count * M * N];

        int x, z;
        for (int f = 0; f < Count * M * N; f++)
        {
            x = f % (M * N);
            z = f / (M * N);
            host_images_R[f] = host_images[z][0][x];
            host_images_G[f] = host_images[z][1][x];
            host_images_B[f] = host_images[z][2][x];
        }
        //Reserva de memoria para los 3 canales
        cudaMalloc((void **)&device_images_R, Count * M * N * sizeof(float));
        cudaMalloc((void **)&device_images_G, Count * M * N * sizeof(float));
        cudaMalloc((void **)&device_images_B, Count * M * N * sizeof(float));
        //Reserva de memoria para el arreglo de salida
        cudaMalloc((void **)&device_images_R_out, M * N * sizeof(float));
        cudaMalloc((void **)&device_images_G_out, M * N * sizeof(float));
        cudaMalloc((void **)&device_images_B_out, M * N * sizeof(float));
        //Pasar datos de canales RGB a VRAM
        cudaMemcpy(device_images_R, host_images_R, Count * M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_images_G, host_images_G, Count * M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_images_B, host_images_B, Count * M * N * sizeof(float), cudaMemcpyHostToDevice);

        float *host_images_R_out_zeros = new float[M * N];
        float *host_images_G_out_zeros = new float[M * N];
        float *host_images_B_out_zeros = new float[M * N];
        for (int f = 0; f < M * N; f++)
        {
            host_images_R_out_zeros[f] = 0;
            host_images_G_out_zeros[f] = 0;
            host_images_B_out_zeros[f] = 0;
        }

        //Inicializamos con 0's los  vector de salida
        cudaMemcpy(device_images_R_out, host_images_R_out_zeros, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_images_R_out, host_images_G_out_zeros, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_images_R_out, host_images_B_out_zeros, M * N * sizeof(float), cudaMemcpyHostToDevice);

        //Medicion de tiempo GPU
        cudaEvent_t ct1, ct2;
        float dt;

        cudaEventCreate(&ct1);
        cudaEventCreate(&ct2);

        cudaEventRecord(ct1);
        kernel<<<grid_size, block_size>>>(Count, M, N, device_images_R, device_images_G, device_images_B, device_images_R_out, device_images_G_out, device_images_B_out);
        cudaEventRecord(ct2);

        cudaEventSynchronize(ct2);
        cudaEventElapsedTime(&dt, ct1, ct2);
       
        std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;

        //Areglos de salida para los resultados de la GPU
        host_image_out = new float *[3];
        for (int i = 0; i < 3; i++)
        {
            host_image_out[i] = new float[M * N];
        }
        // Resultados canal R
        cudaMemcpy(host_image_out[0],device_images_R_out,M * N* sizeof(float),cudaMemcpyDeviceToHost);
        // Resultados canal G
        cudaMemcpy(host_image_out[1], device_images_G_out,M * N*  sizeof(float), cudaMemcpyDeviceToHost);
        // Resultados canal B
        cudaMemcpy(host_image_out[2], device_images_B_out,M * N*  sizeof(float), cudaMemcpyDeviceToHost);
        // Aqui recien usamos el write
        Write(host_image_out, M, N, names[j][2]);
        for (int i = 0; i < 3; i++)
        {
            delete[] host_image_out[i];
        }
        delete[] host_image_out;
        cudaFree(device_images_R);
        cudaFree(device_images_G);
        cudaFree(device_images_B);
        cudaFree(device_images_R_out);
        cudaFree(device_images_G_out);
        cudaFree(device_images_B_out);
    }
    //Free vector de lectura

    return 0;
}
