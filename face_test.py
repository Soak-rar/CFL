

import numpy as np
import matplotlib.pyplot as plt

r = 1.
d = r / np.sqrt(2)
k = 30
width = 20
height = 16
nx = int(width / d) + 1
ny = int(height / d) + 1

occupied = np.zeros((ny, nx))
occupied_coord = np.zeros((ny, nx, 2))
active_list = []
sampled = []

relative = np.array([[-1, 2], [0, 2], [1, 2],
                     [-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1],
                     [-2, 0], [-1, 0], [1, 0], [2, 0],
                     [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
                     [-1, -2], [0, -2], [1, -2]])
np.random.seed(0)
x, y = np.random.rand() * width, np.random.rand() * height
idx_x, idx_y = int(x / d), int(y / d)
occupied[idx_y, idx_x] = 1
occupied_coord[idx_y, idx_x] = (x, y)
active_list.append((x, y))
sampled.append((x, y))

sampled_idx = 0
while len(active_list) > 0:

    idx = np.random.choice(np.arange(len(active_list)))
    ref_x, ref_y = active_list[idx]
    radius = (np.random.rand(k) + 1) * r
    theta = np.random.rand(k) * np.pi * 2
    candidate = radius * np.cos(theta) + ref_x, radius * np.sin(theta) + ref_y
    flag_out = False
    for _x, _y in zip(*candidate):
        if _x < 0 or _x > width or _y < 0 or _y > height:
            continue
        # other geo constraints
        flag = True
        idx_x, idx_y = int(_x / d), int(_y / d)
        if occupied[idx_y, idx_x] != 0:
            continue
        else:
            neighbours = relative + np.array([idx_x, idx_y])
        for cand_x, cand_y in neighbours:
            if cand_x < 0 or cand_x >= nx or cand_y < 0 or cand_y >= ny:
                continue
            if occupied[cand_y, cand_x] == 1:
                cood = occupied_coord[cand_y, cand_x]
                if (_x - cood[0]) ** 2 + (_y - cood[1]) ** 2 < r ** 2:
                    flag = False
                    break
        if flag:
            flag_out = True
            occupied[idx_y, idx_x] = 1
            occupied_coord[idx_y, idx_x] = (_x, _y)
            sampled.append((_x, _y))
            active_list.append((_x, _y))
            sampled_idx += 1
            break
    if not flag_out:
        active_list.pop(idx)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
fig.set_tight_layout(True)
ax.scatter(*zip(*sampled), c='g')
ax.set_xlim([0, width])
ax.set_ylim([0, height])
plt.show()