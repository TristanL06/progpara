import numpy as np
from numba import cuda, float32
from PIL import Image
import argparse
import time
from math import sqrt, ceil


# CUDA kernel functions for image processing
# definition de tous les noyaux dont on aura besoin
gaussian_kernel = np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]], dtype=np.float32)

sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)

sobel_y_kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)


# fonction pour passer en niveaux de gris
@cuda.jit
def rgb_to_bw_kernel(rgb_img, bw_img):
    x, y = cuda.grid(2)
    if x < rgb_img.shape[0] and y < rgb_img.shape[1]:
        R = rgb_img[x, y, 0]
        G = rgb_img[x, y, 1]
        B = rgb_img[x, y, 2]
        bw_img[x, y] = 0.3 * R + 0.59 * G + 0.11 * B

# Fonction de flou gaussien CUDA
@cuda.jit
def gaussian_blur_cuda(input_image, output_image, kernel):
    x, y = cuda.grid(2)
    if x < input_image.shape[0] and y < input_image.shape[1]:
        output_pixel_value = 0.0
        kernel_sum = 0.0  # Variable to store the sum of kernel weights
        for i in range(-2, 3):
            for j in range(-2, 3):
                x_i = x + i
                y_i = y + j
                if x_i >= 0 and x_i < input_image.shape[0] and y_i >= 0 and y_i < input_image.shape[1]:
                    output_pixel_value += input_image[x_i,
                                                      y_i] * kernel[i + 2, j + 2]
                    kernel_sum += kernel[i + 2, j + 2]
        # Normalize by the sum of kernel weights
        output_image[x, y] = output_pixel_value / kernel_sum

# Fonction d'application du filtre de Sobel
@cuda.jit
def sobel_kernel(input_image, output_image):
    x, y = cuda.grid(2)
    if x >= input_image.shape[0] or y >= input_image.shape[1]:
        return
        # Appliquer le filtre de Sobel
    Gx = float32(0.0)
    Gy = float32(0.0)
    for k in range(-1, 2):
        for l in range(-1, 2):
            if x + k > 0 and x + k <= input_image.shape[0] and y + l > 0 and y + l <= input_image.shape[1]:
                Gx += input_image[x + k, y + l] * sobel_x_kernel[k + 1, l + 1]
                Gy += input_image[x + k, y + l] * sobel_y_kernel[k + 1, l + 1]
    output_image[x, y] = min(sqrt(Gx**2 + Gy**2), 175)


@cuda.jit
def threshold_kernel(input_image, threshold_low, threshold_high, output_image):
    x, y = cuda.grid(2)
    if x < input_image.shape[0] and y < input_image.shape[1]:
        if input_image[x, y] < threshold_low / 255:
            output_image[x, y] = 0  # Edge below low threshold
        elif input_image[x, y] > threshold_high / 255:
            output_image[x, y] = 255  # Edge above high threshold
        else:
            output_image[x, y] = 128  # Potential edge between thresholds

@cuda.jit
def hysteresis(input_image, output_image):
    x, y = cuda.grid(2)
    if x >= input_image.shape[0] or y >= input_image.shape[1]:
        return
    if input_image[x, y] > 254:
        output_image[x, y] = 1
        return
    if input_image[x, y] < 1:
        output_image[x, y] = 0
        return
    for k in range(-1, 2):
        for l in range(-1, 2):
            if x + k > 0 and x + k <= input_image.shape[0] and y + l > 0 and y + l <= input_image.shape[1]:
                if input_image[x+k, y+l] > 254:
                    output_image[x, y] = 1
                    return
    output_image[x, y] = 0; 


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="GPU Image Processing")
    parser.add_argument("inputImage", help="Input image filename")
    parser.add_argument("outputImage", help="Output image filename")
    parser.add_argument("--tb", type=int, help="Size of thread block for all operations", default=16)
    parser.add_argument("--bw", action="store_true", help="Perform only the black and white kernel")
    parser.add_argument("--gauss", action="store_true", help="Perform black and white and Gaussian blur kernels")
    parser.add_argument("--sobel", action="store_true", help="Perform all kernels up to Sobel and write magnitude")
    parser.add_argument("--threshold", action="store_true", help="Perform all kernels up to threshold")
    return parser.parse_args()

# Main function
def main():
    args = parse_args()

    # Load input image
    image = Image.open(args.inputImage)
    input_image = np.array(image, dtype=np.float32) / 255.0

    # Allocate memory for output image on GPU
    output_image = np.zeros_like(input_image)

    # Initialize CUDA device arrays
    d_input_image = cuda.to_device(input_image)
    blurred_image = cuda.to_device(input_image)
    
    d_output_image = cuda.to_device(output_image)

    # Define block and grid dimensions
    threads_per_block = (args.tb, args.tb)
    blocks_per_grid_x = (input_image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (input_image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start_time = time.time()

    # Perform selected kernels based on command-line arguments
    if args.bw:
        rgb_to_bw_kernel[blocks_per_grid, threads_per_block](d_input_image, d_output_image)
    elif args.sobel:
        # Implement sobel_kernel and further processing steps
        pass
    elif args.threshold:
        # Implement threshold_kernel and further processing steps
        pass
    else:
        # Perform all kernels
        rgb_to_bw_kernel[blocks_per_grid, threads_per_block](d_input_image, d_output_image)
        gaussian_blur_kernel[blocks_per_grid, threads_per_block](d_output_image, d_input_image, gaussian_kernel)
        # Implement further processing steps for Sobel and thresholding

    # Copy output image from GPU to CPU
    output_image = d_output_image.copy_to_host()

    # Normalize and save output image
    output_image *= 255.0
    output_image = output_image.astype(np.uint8)
    output_image = Image.fromarray(output_image)
    output_image.save(args.outputImage)

    end_time = time.time()
    print("Processing time:", end_time - start_time, "seconds")

if __name__ == "__main__":
    main()
