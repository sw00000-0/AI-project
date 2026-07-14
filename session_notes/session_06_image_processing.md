# Session 06 — Basic Image Processing
**Phase:** AI for Images | **Prereq:** Sessions 01–05 complete

---

## Pre-Class Reading (do before session)

| Resource | What to focus on | Time |
|----------|-----------------|------|
| [OpenCV Python Tutorial — Images Basics](https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html) | Reading, displaying, saving images | 15 min |
| [Pillow (PIL) Tutorial](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html) | First section: open, crop, resize, save | 15 min |
| [What is a pixel? (visual explainer)](https://www.khanacademy.org/computing/pixar/rendering/rendering1/a/rendering-overview) | Understand the pixel grid model | 10 min |
| [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | First 5 minutes — how images become numbers | 10 min |

**Key question to think about before class:**
> "If a computer can only understand numbers, how does it 'see' an image?"

---

## In-Class Agenda

### 1. Images are Just Arrays (10 min)
- A grayscale image = 2D array (height × width), values 0–255
- A colour image = 3D array (height × width × 3 channels: R, G, B)
- Every pixel is a number (or triplet of numbers)
- `0` = black, `255` = white (grayscale)

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Create a simple 8x8 checkerboard manually
checkerboard = np.zeros((8, 8), dtype=np.uint8)
checkerboard[1::2, ::2] = 255   # odd rows, even cols
checkerboard[::2, 1::2] = 255   # even rows, odd cols

plt.imshow(checkerboard, cmap='gray')
plt.colorbar()
plt.title("Checkerboard — I'm just an array!")
plt.show()

print("Shape:", checkerboard.shape)
print("Top-left 4x4:")
print(checkerboard[:4, :4])
```

### 2. Loading and Displaying Real Images (15 min)
```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = Image.open('your_image.jpg')   # or use a URL with requests
img_array = np.array(img)

print("Shape:", img_array.shape)          # (height, width, 3)
print("dtype:", img_array.dtype)          # uint8
print("Min/Max:", img_array.min(), img_array.max())  # 0, 255

# Display with matplotlib
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_array)
axes[0].set_title("Original (RGB)")

# Split channels
axes[1].imshow(img_array[:, :, 0], cmap='Reds')
axes[1].set_title("Red channel")
axes[2].imshow(img_array[:, :, 1], cmap='Greens')
axes[2].set_title("Green channel")
axes[3].imshow(img_array[:, :, 2], cmap='Blues')
axes[3].set_title("Blue channel")
plt.show()
```

Downloading a test image:
```python
import requests
from PIL import Image
from io import BytesIO

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('RGB')
```

### 3. Basic Image Operations with PIL (20 min)
```python
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

img = Image.open('your_image.jpg')

# Resize
img_small = img.resize((200, 200))

# Crop (left, upper, right, lower)
img_crop = img.crop((50, 50, 200, 200))

# Rotate
img_rotated = img.rotate(45, expand=True)

# Flip
img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

# Convert to grayscale
img_gray = img.convert('L')

# Built-in filters
img_blur = img.filter(ImageFilter.GaussianBlur(radius=5))
img_sharpen = img.filter(ImageFilter.SHARPEN)
img_edges = img.filter(ImageFilter.FIND_EDGES)

# Brightness adjustment
enhancer = ImageEnhance.Brightness(img)
img_bright = enhancer.enhance(1.5)   # 1.0 = original, >1 = brighter

# Show side by side
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
imgs = [img, img_gray, img_blur, img_sharpen,
        img_edges, img_rotated, img_flipped, img_bright]
titles = ['Original', 'Grayscale', 'Blur', 'Sharpen',
          'Edge Detect', 'Rotated', 'Flipped', 'Bright']
for ax, im, title in zip(axes.flatten(), imgs, titles):
    ax.imshow(im, cmap='gray' if im.mode == 'L' else None)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### 4. Array-Level Operations (15 min)
Operating directly on the numpy array gives you full control.

```python
img_array = np.array(img)

# Brighten by adding a constant (clip to avoid overflow)
brighter = np.clip(img_array.astype(int) + 50, 0, 255).astype(np.uint8)

# Darken
darker = np.clip(img_array.astype(int) - 50, 0, 255).astype(np.uint8)

# Invert (negative)
inverted = 255 - img_array

# Simple thresholding (grayscale)
gray = np.array(img.convert('L'))
binary = (gray > 128).astype(np.uint8) * 255  # black or white only

# Manual edge detection using difference between neighbours
# Horizontal edges (detect vertical lines)
h_edges = np.abs(gray[:, 1:].astype(int) - gray[:, :-1].astype(int)).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, im, title in zip(axes, [brighter, inverted, binary, h_edges],
                          ['Brightened', 'Inverted', 'Binary', 'H-Edges']):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.show()
```

### 5. Why This Matters for AI (5 min)
- AI image models receive images as these exact numpy arrays
- Data augmentation (flipping, rotating, brightness changes) is how we make models more robust
- Understanding pixel values helps you debug AI model inputs/outputs
- Next sessions: how CNNs learn patterns from these arrays automatically

---

## Practice Problems

### Problem 1 — Image Explorer
Take any photo from your phone or the internet and load it in Python:
1. Print its shape, dtype, min, and max values
2. Display the R, G, B channels separately (as grayscale images)
3. What does each channel tell you about the image? Write 1 sentence per channel.

### Problem 2 — Transformation Grid
Apply all 8 transformations from class (original, grayscale, blur, sharpen, edge detect, rotate, flip, brighten) to the same image. Display them in a 2×4 grid with titles and no axes. Save the figure as a PNG.

### Problem 3 — Manual Brightness Function
Write a function `adjust_brightness(image_path, factor)` that:
- Takes an image path and a factor (0.0 to 2.0, where 1.0 = unchanged)
- Returns a PIL Image with the brightness adjusted
- Handles the case where factor is out of range (clamp to 0–2)
- Do NOT use `ImageEnhance` — do the array math manually

Test it with factor = 0.3, 0.7, 1.0, 1.5, 2.0 and display all results.

### Problem 4 — Pixel Art
Create a 16×16 numpy array and draw something in it using only array assignments (no drawing libraries). For example: a letter, a smiley face, a simple shape. Display it with `plt.imshow()`. Zoom in with `plt.figure(figsize=(6,6))` so it looks crisp.

### Problem 5 — Histogram of Pixel Values
1. Load a colour image
2. Plot histograms of the R, G, B pixel value distributions on the same chart (3 lines, 3 colours)
3. Load a grayscale version of the same image and plot its histogram
4. Compare: what does the pixel distribution tell you about the image?

---

## Vocabulary Added This Session
- Pixel, resolution, aspect ratio
- RGB, grayscale, channel
- Array shape for images: (H, W, C)
- Kernel/filter (blur, sharpen, edge detect)
- Data augmentation
- uint8, clip, overflow
