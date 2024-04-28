import numpy as np
from numba import cuda, float32
from PIL import Image
from math import sqrt, atan2

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
        for i in range(-1, 2):
            for j in range(-1, 2):
                x_i = x + i
                y_i = y + j
                if x_i >= 0 and x_i < input_image.shape[0] and y_i >= 0 and y_i < input_image.shape[1]:
                    output_pixel_value += input_image[x_i, y_i] * kernel[i + 1, j + 1]
                    kernel_sum += kernel[i + 1, j + 1]
        # Normalize by the sum of kernel weights
        output_image[x, y] = output_pixel_value / kernel_sum

# Fonction d'application du filtre de Sobel


@cuda.jit
def sobel_kernel(input_image, output_image):
    i, j = cuda.grid(2)
    if i < input_image.shape[0] - 2 and j < input_image.shape[1] - 2:
        # Appliquer le filtre de Sobel
        Gx = float32(0.0)
        Gy = float32(0.0)
        for k in range(3):
            for l in range(3):
                Gx += input_image[i + k, j + l] * sobel_x_kernel[k, l]
                Gy += input_image[i + k, j + l] * sobel_y_kernel[k, l]
        output_image[i, j] = sqrt(Gx**2 + Gy**2)


# Charger l'image
image = Image.open('input.jpg')

# Allouer la mémoire GPU pour l'image d'entrée et de sortie
input_image = np.array(image, dtype=np.float32) / 255.0
# On ne garde que les deux premières dimensions, car l'image est en niveaux de gris
output_image = np.zeros_like(input_image[:, :, 0], dtype=np.float32)

d_input_image = cuda.to_device(input_image)
d_output_image = cuda.to_device(output_image)

# Définir les dimensions de la grille et des blocs CUDA
threads_per_block = (16, 16)
blocks_per_grid_x = (
    input_image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (
    input_image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)


# Appliquer le filtre noir et blanc en utilisant CUDA
rgb_to_bw_kernel[blocks_per_grid, threads_per_block](
    d_input_image, d_output_image)
# Appliquer le flou gaussien en utilisant CUDA
gaussian_blur_cuda[blocks_per_grid, threads_per_block](
    d_output_image, d_output_image, gaussian_kernel)
# Appliquer le filtre de Sobel en utilisant CUDA
sobel_kernel[blocks_per_grid, threads_per_block](d_output_image, d_output_image)

# Copier l'image de sortie depuis le GPU vers le CPU
output_image = d_output_image.copy_to_host()

# Normalize and convert back to uint8
output_image *= 255.0
output_image = output_image.astype(np.uint8)

# Afficher l'image de sortie
output_image = Image.fromarray(output_image)
output_image.save('output.jpg')
