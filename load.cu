#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "stb/stb_image_resize2.h"

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
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
    int w = width;
    int h = height;
    float fw = w;
    float fh = h;
    float ratio = fh/fw;
    simplify_fraction(&w,&h);
    printf("Aspect ratio: %d/%d\n", h,w);
    //write de nova imagem
    stbi_write_png("img2.png", width, height, channels, img, width * channels);


    //criando imagem menor
    int new_width = 160;
    int new_height = 160*(ratio);
    unsigned char* output = malloc(new_width * new_height * channels);

    stbir_resize_uint8_linear(img, width, height, 0, output, new_width, new_height, 0, channels);
    stbi_write_png("output.png", new_width, new_height, channels, output, new_width * channels);

    stbi_image_free(img);


}