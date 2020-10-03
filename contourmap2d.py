import sys
import numpy as np
import matplotlib.pyplot as plt

file_name = sys.argv[1]
velocity = []
with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
        v = [float(e) for e in line.split(" ")]
        velocity.append(v)

plt.imshow(velocity)
plt.colorbar()
plt.show()