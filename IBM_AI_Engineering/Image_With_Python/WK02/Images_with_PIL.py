# %%
# download the image and set the name
# wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png' -O lenna.png
# wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png' -O baboon.png
# wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png' -O barbara.png


import os

import PIL.Image as Image
# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps

# %%
# set the src , dst dir path
src_dir = os.path.join(os.getcwd(), 'src')
dst_dir = os.path.join(os.getcwd(), 'dst')

# %%
# show the original image
img_lenna = Image.open(os.path.join(src_dir, 'lenna.png'))
type(img_lenna)
img_lenna.show()
# %%
# set the image size and shown by plt
plt.figure(figsize=(10, 10))
plt.imshow(img_lenna)
plt.show()

# %%
print(img_lenna.size)
print(img_lenna.mode)
img_lenna_ld = img_lenna.load()

# %%

x = 0
y = 1
img_lenna_ld[y, x]

img_lenna.save(os.path.join(dst_dir, 'lenna_PIL.jpg'))

# %%
# grayscale the image with PIL
img_lenna_gray = ImageOps.grayscale(img_lenna)
img_lenna_gray.show()
img_lenna_gray.mode

# %%
img_lenna_gray.quantize(256 // 2)
img_lenna_gray.show()


# %%

def get_concat_h(im01, im02):
    # https://note.nkmk.me/en/python-pillow-concat-images/
    dst_img = Image.new("RGB", (im01.width + im02.width, im01.height))
    dst_img.paste(im01, (0, 0))
    dst_img.paste(im02, (im02.width, 0))
    return dst_img


# %%
for n in range(3, 8):
    plt.figure(figsize=(10, 10))
    plt.imshow(get_concat_h(img_lenna_gray, img_lenna_gray.quantize(256 // 2 ** n)))
    plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256 // 2 ** n))
    plt.show()

# %%
# call the src baboon image properties

img_baboon = Image.open(os.path.join(src_dir, 'baboon.png'))
img_baboon

# %%
red, green, blue = img_baboon.split()
get_concat_h(img_baboon, red)
get_concat_h(img_baboon, green)
get_concat_h(img_baboon, blue)

# %%
array = np.array(img_baboon)
print(type(array))
print(array)

array[0, 0]
array.min()
array.max()

# %%

plt.figure(figsize=(10, 10))
plt.imshow(array)
plt.show()

# %%
columns = 256

plt.figure(figsize=(10, 10))
plt.imshow(array[:, 0:columns, :])
plt.show()

# %%
A = array.copy()
plt.imshow(A)
plt.show()
# %%
B = A
A[:, :, :] = 0
plt.imshow(B)
plt.show()

# %%
baboon_array = np.array(img_baboon)
plt.figure(figsize=(10, 10))
plt.imshow(baboon_array)
plt.show()

# %%
baboon_array = np.array(img_baboon)
plt.figure(figsize=(10, 10))
plt.imshow(baboon_array[:, :, :], cmap='gray')
# plt.imshow(baboon_array[:,:,0], cmap='gray')
plt.show()

# %%
baboon_red = baboon_array.copy()
baboon_red[:, :, 0] = 0
baboon_red[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(baboon_red)
plt.show()

# %%
baboon_blue = baboon_array.copy()
baboon_blue[:, :, 0] = 0
baboon_blue[:, :, 1] = 0
plt.figure(figsize=(10, 10))
plt.imshow(baboon_red)
plt.show()
