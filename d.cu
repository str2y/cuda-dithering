#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

const int THREADS_PER_BLOCK = 256;

typedef struct Pixel {
    stbi_uc r;
    stbi_uc g;
    stbi_uc b;
    //stbi_uc a;
} Pixel;

__host__ __device__ void getPixel(const stbi_uc* image, int width, int x, int y, int channels, Pixel* pixel) {
    const stbi_uc* p = image + (channels * (y * width + x));
    pixel->r = p[0];
    pixel->g = p[1];
    pixel->b = p[2];
    //pixel->a = p[3];
}

__host__ __device__ void setPixel(stbi_uc* image, int width, int x, int y, int channels, Pixel* pixel) {
    stbi_uc* p = image + (channels * (y * width + x));
    p[0] = pixel->r;
    p[1] = pixel->g;
    p[2] = pixel->b;
   // p[3] = pixel->a;
}

int gcd(int a, int b) {
    while (b != 0) {
     int temp = b;
     b = a % b;
        a = temp;
 }
 return a;
}

// simplificar fração para o aspect ratio
void simplify_fraction(int *a, int *b) {

int divisor = gcd(*a, *b);

*a /= divisor;
*b /= divisor;

if (*b < 0) {
    *a = (-*a);
    *b = (-*b);
}
}


__global__ void dither(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads){

    int pallete[4][3] =
    {
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

    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id >= total_threads) {
        return;
    }

    int x = thread_id % height;
    int y = thread_id / 160;
    int color,threshold;
    float pixel;
    Pixel inPixel, outPixel;

    getPixel(input_image, width, x, y,channels, &inPixel);
    
    pixel = (inPixel.r*0.2126 + inPixel.g*0.7152 + inPixel.b*0.0722)+(BAYER[x%2][y%2])*.21;
    
    color = floor(pixel/(256/4))>3? 3 : floor(pixel/(256/4));
    
    outPixel.r = pallete[color][0];
    outPixel.g = pallete[color][1];
    outPixel.b = pallete[color][2];

    setPixel(output_image, width, x, y,channels, &outPixel);
    
}


int main(void) {
    int width, height, channels;
    
    stbi_uc *img = stbi_load("img.png", &width, &height, &channels, 0);
    if(img == NULL){
        printf("Error loading\n");
        exit(1);
    }
    printf("Imagem com uma largura de %dpx, altura de %dpx e %d channels\n", width, height, channels);
    int w = width;
    int h = height;
    float fw = w;
    float fh = h;
    float ratio = fh/fw;
    simplify_fraction(&w,&h);
    printf("Aspect ratio: %d/%d\n", h,w);

    //criando imagem menor
    int new_width = 160;
    int new_height = 160*(ratio);
    stbi_uc* output = (stbi_uc *)malloc(new_width * new_height * channels);

    stbir_resize_uint8_linear(img, width, height, 0, output, new_width, new_height, 0, (stbir_pixel_layout)channels);
    stbi_write_png("downscale.png", new_width, new_height, channels, output, new_width * channels);

    stbi_image_free(img);
    
    stbi_uc *ds_img = stbi_load("downscale.png", &width, &height, &channels, 0);

    printf("Imagem com uma largura de %dpx, altura de %dpx e %d channels\n", width, height, channels);

    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* p_input_image;
    stbi_uc* p_output_image;
    stbi_uc* dither_img = (stbi_uc*) malloc(image_size);

    //dither e grayscale
    cudaMallocManaged(&p_input_image, image_size);
    cudaMallocManaged(&p_output_image, image_size);
    cudaMemcpy(p_input_image, ds_img, image_size, cudaMemcpyHostToDevice);

    int total_threads = new_width * new_height;
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / THREADS_PER_BLOCK;
    
    dither<<<blocks, threads>>>(p_input_image, p_output_image, width, height, channels, total_threads);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(dither_img, p_output_image, image_size, cudaMemcpyDeviceToHost);
    stbi_write_png("Ditheredcuda.png", width, height, channels, dither_img, width * channels);
    stbi_image_free(dither_img);
}