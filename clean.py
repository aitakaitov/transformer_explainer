import os


target_dir = "vis/smoothgrad_0.0125_n20_x_inputs"

files = os.listdir(target_dir)

for file in files:
    with open(target_dir + "/" + file, "r", encoding='utf-8') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if lines[i][0] == "{":
            line = lines[i]

    with open(target_dir + "/" + file, "w+", encoding='utf-8') as f:
        f.write(line)

