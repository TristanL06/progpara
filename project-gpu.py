import argparse

def main():
    parser = argparse.ArgumentParser(description="Process images using various filters.")
    parser.add_argument("input_image", help="Input image file path")
    parser.add_argument("output_image", help="Output image file path")
    parser.add_argument("--tb", type=int, help="Threshold value for threshold filter")
    parser.add_argument("--bw", action="store_true", help="Apply black and white filter")
    parser.add_argument("--gauss", action="store_true", help="Apply Gaussian blur filter")
    parser.add_argument("--sobel", action="store_true", help="Apply Sobel filter")
    parser.add_argument("--threshold", action="store_true", help="Apply threshold filter")

    args = parser.parse_args()

    # Now you can access the parsed arguments using args
    print("Input image:", args.input_image)
    print("Output image:", args.output_image)
    print("Threshold value:", args.tb)
    print("Black and white filter:", args.bw)
    print("Gaussian blur filter:", args.gauss)
    print("Sobel filter:", args.sobel)
    print("Threshold filter:", args.threshold)


if __name__ == "__main__":
    main()