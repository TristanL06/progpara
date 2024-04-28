from PIL import Image
import os
import subprocess
import time

def transfer_to_boole(source="", destination=""):
    os.system("scp ./" + source + " boole:projet/" + destination)

def transfer_from_boole(source, destination):
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
transfer_to_boole("input.jpg")
print(execute(["ls -la","python3 projet.py"]))
transfer_from_boole("output.jpg", "/output.jpg")


original = Image.open("input.jpg")
transformed = Image.open("output.jpg")

mixed = Image.new("RGB", (original.width + transformed.width + 5, original.height), (255, 255, 255))
mixed.paste(original, (0, 0))
mixed.paste(transformed, (original.width + 5, 0))
mixed.show()
mixed.save("mixed.jpg")
