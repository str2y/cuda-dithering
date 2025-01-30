#ifndef LOAD_IMAGE_H
#define LOAD_IMAGE_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

const int MAX_THREADS = 1024;
const int THREADS_PER_BLOCK = 256;

typedef struct Pixel {
    stbi_uc r;
    stbi_uc g;
    stbi_uc b;
    stbi_uc a;
} Pixel;


/*__global__ int comparebayer(int a,int b)
{
    if (a >= (b)*.5)
    {
        return 1;
    }
    else if (a >= (b))
    {
        return 2;
    }
    else if (a >= (b)*1.5)
    {
        return 3;
    }
    else
    {
        return 0;
    }
}*/


/*stbi_uc* loadImage(const char* path_to_image, int* width, int* height, int* channels) {
    return stbi_load(path_to_image, width, height, channels, 0);
}*/

void writeImage(const char* path_to_image, stbi_uc* image, int width, int height, int channels) {
    stbi_write_png(path_to_image, width, height, channels, image, width * channels);
}

void imageFree(stbi_uc* image) {
    stbi_image_free(image);
}

/*__host__ __device__ void getPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel) {
    const stbi_uc* col = image + (4 * (y * width + x));
    pixel->r = col[0];
    pixel->g = col[1];
    pixel->b = col[2];
    pixel->a = col[3];
}*/

__host__ __device__ void getPixel(const stbi_uc* image, int width, int x, int y, Pixel* pixel) {
    const stbi_uc* col = image + (4 * (y * width + x));
    pixel->r = col[0];
    pixel->g = col[1];
    pixel->b = col[2];
    pixel->a = col[3];
}

__host__ __device__ void setPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel) {
    stbi_uc* col = image + (4 * (y * width + x));
    col[0] = pixel->r;
    col[1] = pixel->g;
    col[2] = pixel->b;
    col[3] = pixel->a;
}



stbi_uc* gray(stbi_uc* input_image, int width, int height, int channels);
__global__ void grayKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);

stbi_uc* dither(stbi_uc* input_image, int width, int height, int channels);
__global__ void ditherKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);



stbi_uc* gray(stbi_uc* input_image, int width, int height, int channels) {
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height;
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / THREADS_PER_BLOCK;

   
    grayKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void grayKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id >= total_threads) {
        return;
    }

    // Declare coordinates based on thread_id and image dimensions.
    int x_coordinate = thread_id % height;
    int y_coordinate = thread_id / width;

    Pixel inPixel, outPixel;

    getPixel(input_image, width, x_coordinate, y_coordinate, &inPixel);

    int alpha = (inPixel.r*0.216f + inPixel.g*0.7152f + inPixel.b*0.0722f);

    outPixel.r = alpha;
    outPixel.g = alpha;
    outPixel.b = alpha;
    outPixel.a = inPixel.a;

    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel);

}

stbi_uc* dither(stbi_uc* input_image, int width, int height, int channels) {
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = (width * height)/4;
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / THREADS_PER_BLOCK;

   
    ditherKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void ditherKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id >= total_threads) {
        return;
    }

    int pallete[4][3] ={
        { 32, 70, 49},
        { 82,127, 57},
        {174,196, 64},
        {215,232,148}
    };
    
    int BAYER[2][2] =   
    { 
    {0,127}, 
    {191,63}
    };
    
    // Declare coordinates based on thread_id and image dimensions.
    int x_coordinate = (thread_id*2) % height;
    int y_coordinate = (thread_id*2) / width;

    Pixel Pixel1,Pixel2,Pixel3,Pixel4,outPixel1,outPixel2,outPixel3,outPixel4;
    int p1,p2,p3,p4;

    getPixel(input_image, width, x_coordinate,   y_coordinate,   &Pixel1);
    getPixel(input_image, width, x_coordinate+1, y_coordinate,   &Pixel2);
    getPixel(input_image, width, x_coordinate,   y_coordinate+1, &Pixel3);
    getPixel(input_image, width, x_coordinate+1, y_coordinate+1, &Pixel4);
    
    //p1
    
    if (Pixel1.r >= (BAYER[0][0])*.5)
    {
        p1=1;
    }
    else if (Pixel1.r >= (BAYER[0][0]))
    {
       p1=2;
    }
    else if (Pixel1.r >= (BAYER[0][0])*1.5)
    {
        p1=3;
    }
    else
    {
       p1=0;
    }
    
    //p2
    
       if (Pixel2.r >= (BAYER[1][0])*.5)
    {
        p2=1;
    }
    else if (Pixel2.r >= (BAYER[1][0]))
    {
       p2=2;
    }
    else if (Pixel2.r >= (BAYER[1][0])*1.5)
    {
        p2=3;
    }
    else
    {
       p2=0;
    }

    
    //p3
    
    
    if (Pixel1.r >= (BAYER[0][1])*.5)
    {
        p3=1;
    }
    else if (Pixel3.r >= (BAYER[0][1]))
    {
       p1=2;
    }
    else if (Pixel3.r >= (BAYER[0][1])*1.5)
    {
        p1=3;
    }
    else
    {
       p1=0;
    }

    //p4

    if (Pixel4.r>= (BAYER[1][1])*.5)
    {
        p4=1;
    }
    else if (Pixel4.r >= (BAYER[1][1]))
    {
       p4=2;
    }
    else if (Pixel4.r >= (BAYER[1][1])*1.5)
    {
        p4=3;
    }
    else
    {
       p4=0;
    }
    
    

    /*
    int p1 = comparebayer(Pixel1.r,BAYER[0][0]);
    int p2 = comparebayer(Pixel2.r,BAYER[1][0]);
    int p3 = comparebayer(Pixel3.r,BAYER[0][1]);
    int p4 = comparebayer(Pixel3.r,BAYER[1][1]);
    */
    
    //set pixel 1
    outPixel1.r = pallete[p1][0];
    outPixel1.g = pallete[p1][1];
    outPixel1.b = pallete[p1][2];
    outPixel1.a = 255;

    //set pixel 2
    outPixel2.r = pallete[p2][0];
    outPixel2.g = pallete[p2][1];
    outPixel2.b = pallete[p2][2];
    outPixel2.a = 255;

    //set pixel 2
    outPixel3.r = pallete[p3][0];
    outPixel3.g = pallete[p3][1];
    outPixel3.b = pallete[p3][2];
    outPixel3.a = 255;

    //set pixel 2
    outPixel4.r = pallete[p4][0];
    outPixel4.g = pallete[p4][1];
    outPixel4.b = pallete[p4][2];
    outPixel4.a = 255;
    
    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel1);
    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel2);
    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel3);
    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel4);
}

#endif