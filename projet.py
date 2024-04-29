import numpy as np
from numba import cuda, float32
from PIL import Image
from math import sqrt, ceil

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
    gx = float32(0.0)
    gy = float32(0.0)
    for k in range(-1, 2):
        for l in range(-1, 2):
            if x + k > 0 and x + k <= input_image.shape[0] and y + l > 0 and y + l <= input_image.shape[1]:
                gx += input_image[x + k, y + l] * sobel_x_kernel[k + 1, l + 1]
                gy += input_image[x + k, y + l] * sobel_y_kernel[k + 1, l + 1]
    output_image[x, y] = min(sqrt(gx**2 + gy**2), 175)


@cuda.jit
def threshold_kernel(input_image, threshold_low, threshold_high, output_image):
    x, y = cuda.grid(2)
    if x < input_image.shape[0] and y < input_image.shape[1]:
        if input_image[x, y] < threshold_low / 255:
            output_image[x, y] = 0  # Edge below low threshold
        elif input_image[x, y] > threshold_high / 255:
            output_image[x, y] = 1  # Edge above high threshold
        else:
            output_image[x, y] = 0.5  # Potential edge between thresholds

@cuda.jit
def hysteresis(input_image, output_image):
    x, y = cuda.grid(2)
    if x >= input_image.shape[0] or y >= input_image.shape[1]:
        return
    if input_image[x, y] == 1:
        output_image[x, y] = 1
        return
    if input_image[x, y] < 1:
        output_image[x, y] = 0
        return
    for k in range(-1, 2):
        for l in range(-1, 2):
            if x + k > 0 and x + k <= input_image.shape[0] and y + l > 0 and y + l <= input_image.shape[1]:
                if input_image[x+k, y+l] == 1:
                    output_image[x, y] = 1
                    return
    output_image[x, y] = 0; 

# Function to compute the number of thread blocks
def compute_thread_blocks(imagetab, block_size):
    """
    Computes the number of thread blocks required for CUDA operations.

    Args:
        imagetab (numpy.ndarray): Input image as a NumPy array.
        block_size (tuple): Size of the thread block in (height, width) format.

    Returns:
        tuple: Number of thread blocks required in (blockspergrid_y, blockspergrid_x) format.
    """
    height, width = imagetab.shape[:2]
    blockspergrid_x = ceil(width / block_size[0])
    blockspergrid_y = ceil(height / block_size[1])
    blockspergrid = (blockspergrid_y, blockspergrid_x)
    return blockspergrid

# Charger l'image
image = Image.open('input.jpg')

# Allouer la mémoire GPU pour l'image d'entrée et de sortie
input_image = np.array(image, dtype=np.float32) / 255
# On ne garde que les deux premières dimensions, car l'image est en niveaux de gris
gl = np.zeros_like(input_image[:, :, 0], dtype=np.float32)
blurred = np.zeros_like(input_image[:, :, 0], dtype=np.float32)
sobeled = np.zeros_like(input_image[:, :, 0], dtype=np.float32)
thresoled = np.zeros_like(input_image[:, :, 0], dtype=np.float32)
hysteresised = np.zeros_like(input_image[:, :, 0], dtype=np.float32)

d_input_image = cuda.to_device(input_image)
gl_image = cuda.to_device(gl)
blurred_image = cuda.to_device(blurred)
sobeled_image = cuda.to_device(sobeled)
thresoled_image = cuda.to_device(thresoled)
hysteresised_image = cuda.to_device(hysteresised)

# Définir les dimensions de la grille et des blocs CUDA
threads_per_block = (16, 16)
blocks_per_grid = compute_thread_blocks(input_image, threads_per_block)


# Appliquer le filtre noir et blanc
rgb_to_bw_kernel[blocks_per_grid, threads_per_block](d_input_image, gl_image)
# Appliquer le flou gaussien
gaussian_blur_cuda[blocks_per_grid, threads_per_block](gl_image, blurred_image, gaussian_kernel)
# Appliquer le filtre de Sobel
sobel_kernel[blocks_per_grid, threads_per_block](blurred_image, sobeled_image)
# appliquer le filtre de seuillage
threshold_kernel[blocks_per_grid, threads_per_block](sobeled_image, 51, 102, thresoled_image)
# appliquer le filtre à hystérésis
hysteresis[blocks_per_grid, threads_per_block](thresoled_image, hysteresised_image)

# Copier l'image de sortie depuis le GPU vers le CPU
out1 = gl_image.copy_to_host()
out2 = blurred_image.copy_to_host()
out3 = sobeled_image.copy_to_host()
out4 = thresoled_image.copy_to_host()
out5 = hysteresised_image.copy_to_host()

images = [out1, out2, out3, out4, out5]
# Sauvegarder l'image de sortie
for i in range(len(images)):
    images[i] *= 255.0
    images[i] = images[i].astype(np.uint8)
    output_image = Image.fromarray(images[i])
    output_image.save(f'output_{i}.jpg')