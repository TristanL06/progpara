from PIL import Image
import os

image = Image.new("RGB", (11, 11))

px = image.load()

for i in range(11):
    for j in range(11):
        px[i, j] = (0, 0, 0) if (i+j)%2 == 0 else (255, 255, 255)


image.save("progParallèle/test.jpg")

os.system("scp progParallèle/input.jpg lt925100@boole.polytech.unice.fr:/net/home/l/lt925100/projet/input.jpg")

wait = input("Press Enter to continue...")

os.system("scp lt925100@boole.polytech.unice.fr:/net/home/l/lt925100/projet/output.jpg progParallèle/")

image = Image.open("progParallèle/output.jpg")

original = Image.open("progParallèle/input.jpg")
transformed = Image.open("progParallèle/output.jpg")

mixed = Image.new("RGB", (original.width + transformed.width + 5, original.height), (255, 255, 255))
mixed.paste(original, (0, 0))
mixed.paste(transformed, (original.width + 5, 0))
mixed.show()
mixed.save("progParallèle/mixed.jpg")