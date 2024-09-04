# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageOps

# %%
# get current work dir
# set the set, dst folder path

cwd = os.getcwd()
src_dir = os.path.join(cwd, 'src')
dst_dir = os.path.join(cwd, 'dst')

# %%

img_baboon = np.array(Image.open(os.path.join(src_dir, 'baboon.png')))
plt.figure(figsize=(5, 5))
plt.imshow(img_baboon)
plt.show()

# %%
A = img_baboon
id(A) == id(img_baboon)

# memory address are diiferent whene we use method copy()
B = img_baboon.copy()
id(B) == id(img_baboon)

# %%
img_baboon[:, :, ] = 0
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(img_baboon)
plt.title('baboon')
plt.subplot(122)
plt.imshow(A)
plt.title('array A')
plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(img_baboon)
plt.title('baboon')
plt.subplot(122)
plt.imshow(B)
plt.title('array B')
plt.show()

# %%
img_cat = Image.open(os.path.join(src_dir, 'cat.png'))
plt.figure(figsize=(10, 10))
plt.imshow(img_cat)
plt.show()

# %%
array = np.array(img_cat)
width, height, C = array.shape
print('width, height, C', width, height, C)

# %%
array_flip = np.zeros((width, height, C), dtype=np.uint8)

for i, row in enumerate(array):
    array_flip[width - 1 - i, :, :] = row

# %%
img_flip = ImageOps.flip(img_cat)
plt.figure(figsize=(5, 5))
plt.imshow(img_flip)
plt.show()

img_mirror = ImageOps.mirror(img_cat)
plt.figure(figsize=(5, 5))
plt.imshow(img_mirror)
plt.show()

# %%

img_flip = img_cat.transpose(1)
plt.imshow(img_flip)
plt.show()

# %%
flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE,
        "TRANSVERSE": Image.TRANSVERSE}

flip["FLIP_LEFT_RIGHT"]

# %%
for key, values in flip.items():
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img_cat)
    plt.subplot(1, 2, 2)
    plt.imshow(img_cat.transpose(values))
    plt.title(key)
    plt.show()

# %%
upper = 150
lower = 400
crop_top = array[upper: lower, :, :]
plt.figure(figsize=(5, 5))
plt.imshow(crop_top)
plt.show()

# %%
left = 150
right = 400
crop_horizontal = crop_top[:, left:right, :]
plt.figure(figsize=(5, 5))
plt.imshow(crop_horizontal)
plt.show()

# %%

crop_img_cat = img_cat.crop((left, upper, right, lower))
plt.figure(figsize=(5, 5))
plt.imshow(crop_img_cat)
plt.show()

# %%
crop_img_cat = crop_img_cat.transpose(Image.FLIP_LEFT_RIGHT)
crop_img_cat

# %%
array_S
