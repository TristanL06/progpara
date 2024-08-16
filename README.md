# EIENPF8 - Programmation Fonctionnelle et parallèle
### Polytech Nice Sophia 2023-2024, filière Science Informatique
### Projet Canny Edge détection par groupe de 2 personnes

utiliser le script "test.py" pour exécuter le projet sur la VM boole interne à Polytech. Pour ce faire il est nécessaire de configurer une clé SSH et une configuration de profile.
Les étapes pour faire cela sont détaillées dans ce repo : [https://github.com/lieutenantX/Projet_Pi/blob/main/ressources/configure_ssh.md](https://github.com/lieutenantX/Projet_Pi/blob/main/ressources/configure_ssh.md)

---
# Sujet original (en anglais)
This is a **2-person** project. The goal of this project is to implement a Canny Edge Detector in Cuda.

![valve](https://raw.githubusercontent.com/TristanL06/progpara/master/valve.png)
![Valve monochrome canny](https://raw.githubusercontent.com/TristanL06/progpara/master/Valve_monochrome_canny.png)

## Description of the algoritm
The algorithm works on RGB images in multiple steps. Each step can be implemented separately, although they are all linked in the end

1. Convert the original image to a grayscale version (bw_kernel)
2. Apply a Gaussian blur (gaussian_kernel)
3. Compute the Sobel magnitude and angle of each pixel (sobel_kernel)
4. Find potential edges  based on thresholds   (threshold_kernel)
5. Suppress edges which are too weak (hysterisis_kernel)
A detailed implementation of the kernels can be found in the paper *Canny Edge Detection on GPU using CUDA* at the bottom of this page.

## Project
The project should be submitted as a unique *project-gpu.py* file. Your program will take as an input an image to process and write the resulting image on disk. It should support the following usage
```sh
python project-gpu.py [--tb int] [--bw] [--gauss] [--sobel] [--threshold]  <inputImage> <outputImage>
```
with
- inputImage :  the source image
- outputImage : the destination image  
- --tb int : optional size of a thread block for all operations
- --bw : perform only the bw_kernel
- --gauss : perform the bw_kernel and the gauss_kernel 
- --sobel : perform all kernels up to sobel_kernel  and write to disk the magnitude of each pixel
- --threshold : perform all kernels up to threshold_kernel
If there is no optional argument about kernels, the program will perform all kernels and produce the final image. 

The evaluation will be performed using automated tests in steps of increasing difficulty and rewards and use the following criteria

- Success/failures of the tests
- Code quality (naming convention, useful comments)
- The performance of your code as measured with time

## Q&A
- *What formula to use to convert RGB to BW ?* You can use the one from the first lab session
- *What Gaussian kernel to use ?* The one given at the beginning of the second lab session
- *What are good values for the thresholds ?* Papers use  51 for the low one, and  102 for the high one. Use those in your project
- *The Sobel part gives values larger than 255, what should I do?*  The paper at the bottom suggest clamping the values to 175 (Fig. 5) 
- *How do I know my code is correct ?* Test each step individually, both visually (BW and Blur) and using well constructed arrays so you can check their content manually \
- *It looks hard...* The most complicated step is the hysteresis phase, which is not trivial to implement with Cuda.  
- *Any tip ?* There are a couple of C/C++ implementation you can look at on github. Be aware that they don't implement the steps in the same way or even using the same algorithms. 

## Resources
- A detailled Python implementation : [https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123)
- An implementation using OpenCV : [https://indiantechwarrior.com/canny-edge-detection-for-image-processing/](https://indiantechwarrior.com/canny-edge-detection-for-image-processing/)
- A CUDA/C implementation (not tested, probably working) : [https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/CannyGPU.cu](https://github.com/Dantekk/Canny-GPU-CUDA-implementation/blob/main/CannyGPU.cu)
