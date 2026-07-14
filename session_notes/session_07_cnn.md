# Session 07 — Convolutional Neural Networks (CNNs)
**Phase:** AI for Images | **Prereq:** Sessions 01–06 complete

---

## Pre-Class Reading (do before session)

| Resource | What to focus on | Time |
|----------|-----------------|------|
| [3Blue1Brown — But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) | Watch fully — best visual explanation of convolution | 23 min |
| [CS231n — CNNs for Visual Recognition (architecture section only)](https://cs231n.github.io/convolutional-networks/#architectures) | Read "Architecture Overview" and "Case Study" sections | 20 min |
| [Towards Data Science — CNNs Explained](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) | Good visual walkthrough — focus on the diagrams | 15 min |

**Key question to think about before class:**
> "How does a computer learn to recognise a cat in a photo — without being told what makes a cat a cat?"

---

## In-Class Agenda

### 1. From Pixels to Predictions — The Big Picture (10 min)
- Regular neural networks flatten an image into one long vector → loses spatial relationships
- CNNs preserve spatial structure — nearby pixels stay nearby
- Core idea: learn *filters* that detect patterns (edges, textures, shapes) automatically

Three key operations:
1. **Convolution** — slide a small filter over the image to detect a pattern
2. **Pooling** — reduce the spatial size (summarise and shrink)
3. **Fully connected** — combine all learned features to make a final prediction

### 2. Convolution — What Actually Happens (15 min)
A filter (kernel) is a small matrix (e.g., 3×3) of learnable numbers. It slides over the image and produces a new image (feature map).

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Manual convolution to show what filters detect
def apply_filter(image_gray, kernel):
    """Apply a 3x3 filter manually using numpy."""
    H, W = image_gray.shape
    output = np.zeros((H-2, W-2))
    for i in range(H-2):
        for j in range(W-2):
            patch = image_gray[i:i+3, j:j+3]
            output[i, j] = np.sum(patch * kernel)
    return np.clip(output, 0, 255)

# Famous filters
edge_horizontal = np.array([[-1, -1, -1],
                              [ 0,  0,  0],
                              [ 1,  1,  1]])

edge_vertical   = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]])

sharpen         = np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]])

blur            = np.ones((3,3)) / 9.0

# Load and convert to grayscale
img = np.array(Image.open('your_image.jpg').convert('L').resize((300, 300)))

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
titles = ['Original', 'H-Edges', 'V-Edges', 'Sharpen', 'Blur']
results = [img,
           apply_filter(img, edge_horizontal),
           apply_filter(img, edge_vertical),
           apply_filter(img, sharpen),
           apply_filter(img, blur)]

for ax, im, t in zip(axes, results, titles):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
    ax.axis('off')
plt.show()
```

**Key insight:** In a CNN, the model *learns* these filter values — it figures out which patterns help with the task.

### 3. Running a Pre-trained CNN (20 min)

Using torchvision's ResNet — already trained on ImageNet (1000 categories):

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained ResNet (downloads weights automatically first time)
model = models.resnet50(pretrained=True)
model.eval()   # set to inference mode

# Image preprocessing (ResNet expects specific input format)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load an image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img = Image.open(BytesIO(requests.get(url).content)).convert('RGB')

# Preprocess and predict
input_tensor = preprocess(img).unsqueeze(0)   # add batch dimension
with torch.no_grad():
    output = model(input_tensor)

# Get top 5 predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_idx = torch.topk(probabilities, 5)

# Load ImageNet class names
import json
labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(labels_url).text)

print("Top 5 predictions:")
for prob, idx in zip(top5_prob, top5_idx):
    print(f"  {labels[idx]}: {prob.item():.2%}")
```

### 4. Popular CNN Architectures — The Family Tree (10 min)

| Architecture | Year | Key Innovation |
|-------------|------|---------------|
| LeNet | 1998 | First practical CNN — handwriting recognition |
| AlexNet | 2012 | Deep CNN won ImageNet — started the deep learning era |
| VGG | 2014 | Very deep (16-19 layers), simple design |
| ResNet | 2015 | **Residual connections** — solved vanishing gradient for very deep nets |
| EfficientNet | 2019 | Scales width/depth/resolution together efficiently |
| Vision Transformer (ViT) | 2020 | Applies Transformer (from NLP) to images — now state of the art |

**For most practical projects:** use ResNet50 or EfficientNet as your starting point.

### 5. Using HuggingFace for Image Classification (10 min)
```python
from transformers import pipeline

# Much simpler! Pipeline handles preprocessing
image_classifier = pipeline("image-classification",
                             model="google/vit-base-patch16-224")

# Classify an image from URL
results = image_classifier("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")

for r in results[:5]:
    print(f"{r['label']}: {r['score']:.2%}")
```

---

## Practice Problems

### Problem 1 — Filter Explorer
Take any image and apply all 4 filters from class (horizontal edges, vertical edges, sharpen, blur).
1. Display all 5 (original + 4 filters) side by side
2. Describe in one sentence what each filter detects/does to the image
3. What happens if you apply the blur filter twice? Three times?

### Problem 2 — ResNet Classification
Use the ResNet50 code from class to classify 5 different images (find them online):
1. Two animals of different species
2. One food item
3. One vehicle
4. One image of your choice

For each, print the top 3 predictions and their probabilities. Did the model get them right?

### Problem 3 — HuggingFace vs ResNet Comparison
Take the same 3 images from Problem 2 and classify them using both:
- The manual ResNet50 pipeline from class
- The HuggingFace `image-classification` pipeline

Do they agree? Which one gives more useful/descriptive labels?

### Problem 4 — Visualise What CNNs See
Using the ResNet model, extract and visualise the output of the first convolutional layer:
```python
# Hint: register a hook to capture intermediate activations
activation = {}
def hook_fn(module, input, output):
    activation['layer1'] = output.detach()

model.layer1.register_forward_hook(hook_fn)
# Run inference, then visualise activation['layer1']
```
Display at least 16 of the feature maps. What do they look like compared to the original image?

### Problem 5 — Reflection
In your own words (no more than 150 words), explain:
> "What is the difference between a traditional program (like if/else rules) and a CNN for recognising images? Why would you use a CNN instead of writing a list of rules?"

---

## Vocabulary Added This Session
- Convolutional Neural Network (CNN)
- Filter / kernel, feature map, stride, padding
- Pooling (max pooling, average pooling)
- Fully connected layer
- Activation function (ReLU)
- Depth, width (of a network)
- Vanishing gradient, residual connection
- ImageNet, top-1 accuracy, top-5 accuracy
- Batch, batch dimension
- Inference mode (`model.eval()`)
