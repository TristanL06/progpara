from PIL import Image, ImageDraw, ImageFont
import os
import subprocess
import time

input = "input.jpg"

def transfer_to_boole(source="", destination=""):
    if os.path.isfile(source):
        os.system("scp ./" + source + " boole:projet/" + destination)
    elif os.path.isdir(source):
        os.system("scp -r ./" + source + " boole:projet/" + destination)

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

transfer_to_boole("project-gpu.py")

transfer_to_boole("projet.py")
transfer_to_boole(input, "input.jpg")
transfer_to_boole("env.py")
transfer_to_boole("benchmark.py")
transfer_to_boole("img", "img")

#print(execute(["python3 benchmark.py img output --tb 16"]))
#print(execute(["python3 projet.py"]))

print(execute([
    "python3 project-gpu.py input.jpg output1_bw.jpg --bw",
    "python3 project-gpu.py input.jpg output2_gauss.jpg --gauss",
    "python3 project-gpu.py input.jpg output3_sobel.jpg --sobel",
    "python3 project-gpu.py input.jpg output4_threshold.jpg --threshold",
    "python3 project-gpu.py input.jpg output5_final.jpg"]))

print(execute([
    "python3 project-gpu.py input.jpg output1_bw.jpg --tb 16 --bw",
    "python3 project-gpu.py input.jpg output2_gauss.jpg --tb 16 --gauss",
    "python3 project-gpu.py input.jpg output3_sobel.jpg --tb 16 --sobel",
    "python3 project-gpu.py input.jpg output4_threshold.jpg --tb 16 --threshold",
    "python3 project-gpu.py input.jpg output5_final.jpg --tb 16"]))

transfer_from_boole("output*")
print(execute(["rm -r output*"]))

images = [f for f in os.listdir() if f.startswith("output")]

input_image = Image.open(input)
x, y = input_image.size
mixed = Image.new("RGB", ((5 + x) * (len(images)+1) - 5, y +x//12) , (255, 255, 255))
mixed.paste(input_image, (0, 0))
draw = ImageDraw.Draw(mixed)
font = ImageFont.load_default(size=x/20)
text_size = draw.textlength("Image originale", font=font)
draw.text(((x-text_size)/2, y), "Image originale", (0, 0, 0), font=font)
# compile all images in one
for i in range(len(images)):
    text = images[i].split("_")[1].split(".")[0]
    mixed.paste(Image.open(images[i]), ((x + 5)*(i+1), 0))
    text_size = draw.textlength(text, font=font)
    draw.text((x + 5 + (x + 5)*i + (x-text_size)/2, y), text, (0, 0, 0), font=font)
    

mixed.save("mixed.jpg")

mixed = Image.open("mixed.jpg")
mixed.show()
