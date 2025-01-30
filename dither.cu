#include <stdio.h>
#include <string.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "image.h"

   int gcd(int a, int b) {
    while (b != 0) {
     int temp = b;
     b = a % b;
        a = temp;
 }
 return a;
}

// Function to simplify a fraction
void simplify_fraction(int *a, int *b) {
if (*b == 0) {
   printf("w cannot be zero.\n");
    return;
}

int divisor = gcd(*a, *b);

*a /= divisor;
*b /= divisor;

if (*b < 0) { // Normalize the sign
    *a = (-*a);
    *b = (-*b);
}
}

int main(void){
int width, height, channels;
unsigned char *img = stbi_load("img.png", &width, &height, &channels, 0);
if(img == NULL){
    printf("Error loading\n");
    exit(1);
}
//printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);


int w = width;
int h = height;
float fw = w;
float fh = h;
float ratio = fh/fw;
simplify_fraction(&w,&h);

stbi_uc* filtered_image;
filtered_image = gray(img, width, height, channels);
writeImage("gray.png", filtered_image, width, height, channels);
imageFree(filtered_image);
//
//printf("Aspect ratio: %d/%d\n", h,w);
//write de nova imagem
//stbi_write_png("img2.png", width, height, channels, img, width * channels);


//criando imagem menor
int new_width = 160;
int new_height = 160*ratio;
unsigned char *output = (unsigned char*)malloc(new_width * new_height * 4);

stbir_pixel_layout ch = (stbir_pixel_layout)channels;
stbir_resize_uint8_linear(img, width, height, 0, output, new_width, new_height, 0, ch);
stbi_write_png("downscale.png", new_width, new_height, channels, output, new_width * channels);
imageFree(img);

unsigned char *gs_img = stbi_load("downscale.png", &width, &height, &channels, 0);
stbi_uc* dithered_image;
dithered_image = dither(gs_img, width, height, channels);
writeImage("dither.png", dithered_image, new_width, new_height, channels);

stbi_image_free(gs_img);



}