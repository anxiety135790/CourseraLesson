# %%
# download the image and set the name
# wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png' -O lenna.png
# wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png' -O baboon.png
# wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png' -O barbara.png


# %%
import os

import cv2
import matplotlib.pyplot as plt
from PIL.Image import Image


# %%
def get_concat_h(im01, im02):
    dst = (Image.new('RGB', (im01.width + im02.width), im01.height))
    dst.paste(im01, (0, 0))
    dst.paste(im02, (im01.width, 0))
    return


# %%
# get current work dir
# set the set, dst folder path

cwd = os.getcwd()
src = os.path.join(cwd, 'src')
dst = os.path.join(cwd, 'dst')

# %%
# read, set and show image
lenna = cv2.imread(os.path.join(src, 'lenna.png'))
type(lenna)
plt.figure(figsize=(10, 10))
plt.imshow(lenna)
plt.show()

# %%
# change the color space
lenna_rgb = cv2.cvtColor(lenna, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(lenna_rgb)
plt.show()

# %%
# grayscale Images

lenna_gray = cv2.cvtColor(lenna_rgb, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10, 10))
plt.imshow(lenna_gray, cmap='gray')
plt.show()
cv2.imwrite(os.path.join(dst, 'lenna_OpenCV.jpg'), lenna_gray)

# %%
# grayscale barbara

barbara_gray = cv2.imread(os.path.join(src, 'barbara.png'), cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 10))
plt.imshow(barbara_gray, cmap='gray')
plt.show()

# %%
# Color Channel

baboon = cv2.imread(os.path.join(src, 'baboon.png'))
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

# %%
# concatenate the baboon in blue, green, red
blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]
im_bgr = cv2.vconcat([blue, green, red])

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("RGB image")
plt.subplot(121)
plt.imshow(im_bgr, cmap='gray')
plt.title("Different color blue,gerren,red")
plt.show()

# %%
rows = 256
plt.figure(figsize=(10, 10))
plt.imshow(lenna[0: rows, :, :])
plt.show()

columns = 256
A = lenna.copy()
plt.imshow(A[:, 0: columns, :])
plt.show()

# %%
B = A
A[:, :, :] = 0
plt.imshow(B)
plt.show()

# %%
baboo_red = baboon.copy()
baboo_red[:, :, 0] = 0
baboo_red[:, :, 1] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboo_red, cv2.COLOR_BGR2RGB))
plt.show()

baboo_blue = baboon.copy()
baboo_blue[:, :, 1] = 0
baboo_blue[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboo_blue, cv2.COLOR_BGR2RGB))
plt.show()

baboo_green = baboon.copy()
baboo_green[:, :, 0] = 0
baboo_green[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboo_green, cv2.COLOR_BGR2RGB))
plt.show()
