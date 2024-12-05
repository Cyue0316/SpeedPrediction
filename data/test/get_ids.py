
with open("data_output.txt", "r") as f:
    for line in f.readlines():
        linkids = line.split(",")
        
print(f"{linkids[152]},{linkids[252]}")