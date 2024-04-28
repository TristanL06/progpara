from PIL import Image
import os
import subprocess
import time

input = "valve.png"

def transfer_to_boole(source="", destination=""):
    os.system("scp ./" + source + " boole:projet/" + destination)

def transfer_from_boole(source, destination = ""):
    os.system("scp boole:projet/" + source + " ./" + destination)

def execute(commands):
    results = ["Results:"]
    for command in commands:
        try:
            # Utilisation de subprocess.check_output() pour obtenir la sortie de la commande
            output = subprocess.check_output(["ssh", "boole", f'cd projet && {command}'], universal_newlines=True)
            results.append(output)
        except subprocess.CalledProcessError as e:
            results.append(f"Error executing command: {e.output}")
    results.append("====================================")
    return results

transfer_to_boole("projet.py")
transfer_to_boole(input, "input.jpg")
print(execute(["ls -la","python3 projet.py"]))
transfer_from_boole("out*")

images = [f for f in os.listdir() if f.startswith("output_")]

input_image = Image.open(input)
mixed = Image.new("RGB", ((5 + input_image.size[0]) * len(images), input_image.size[1]) , (255, 255, 255))
mixed.paste(input_image, (0, 0))
# compile all images in one
for i in range(len(images)):
    mixed.paste(Image.open(images[i]), ((input_image.size[0] + 5)*i - 5, 0))
mixed.save("mixed.jpg")

mixed = Image.open("mixed.jpg")
mixed.show()
