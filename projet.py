import numpy as np
from numba import cuda
from PIL import Image

# Définir le noyau gaussien
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32)

gaussian_kernel2 = np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]], dtype=np.float32)

# Fonction de flou gaussien CUDA
@cuda.jit
def gaussian_blur_cuda(input_image, output_image, kernel):
    x, y = cuda.grid(2)
    if x < input_image.shape[0] and y < input_image.shape[1]:
        for c in range(3):  # Loop over color channels
            output_pixel_value = 0.0
            kernel_sum = 0.0  # Variable to store the sum of kernel weights
            for i in range(-1, 2):
                for j in range(-1, 2):
                    x_i = x + i
                    y_i = y + j
                    if x_i >= 0 and x_i < input_image.shape[0] and y_i >= 0 and y_i < input_image.shape[1]:
                        output_pixel_value += input_image[x_i, y_i, c] * kernel[i + 1, j + 1]
                        kernel_sum += kernel[i + 1, j + 1]
            output_image[x, y, c] = output_pixel_value / kernel_sum  # Normalize by the sum of kernel weights

# Charger l'image
image = Image.open('input.jpg')

# Allouer la mémoire GPU pour l'image d'entrée et de sortie
input_image = np.array(image, dtype=np.float32) / 255.0
output_image = np.zeros_like(input_image)

d_input_image = cuda.to_device(input_image)
d_output_image = cuda.to_device(output_image)

# Définir les dimensions de la grille et des blocs CUDA
threads_per_block = (16, 16)
blocks_per_grid_x = (input_image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (input_image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Appliquer le flou gaussien en utilisant CUDA
gaussian_blur_cuda[blocks_per_grid, threads_per_block](d_input_image, d_output_image, gaussian_kernel2)

# Copier l'image de sortie depuis le GPU vers le CPU
output_image = d_output_image.copy_to_host()

# Normalize and convert back to uint8
output_image *= 255.0
output_image = output_image.astype(np.uint8)

# Afficher l'image de sortie
output_image = Image.fromarray(output_image)
output_image.save('output.jpg')
output_image.show()
