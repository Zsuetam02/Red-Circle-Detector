import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === LOAD IMAGES ===

image_folder = r'C:\Users\matsk\PyCharmMiscProject\AOLab2\data'
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.bmp')]
images = [cv2.imread(os.path.join(image_folder, f)) for f in image_files]
num_images = len(images)

# === CONVERT TO HSV ===

hsv_images = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in images]
hue = [img[:, :, 0] / 179.0 for img in hsv_images]
sat = [img[:, :, 1] / 255.0 for img in hsv_images]
val = [img[:, :, 2] / 255.0 for img in hsv_images]

# === DETECT RED REGIONS & MEASURE AREA ===

clean_masks = []
areas = []

for i in range(num_images):
    h, s, v = hue[i], sat[i], val[i]

    # thresholds
    red_mask1 = ((h >= 0.0) & (h <= 0.05)) & (s > 0.35) & (v > 0.15)
    red_mask2 = ((h >= 0.95) & (h <= 1.0)) & (s > 0.35) & (v > 0.15)
    combined_mask = np.logical_or(red_mask1, red_mask2).astype(np.uint8)

    # fill holes
    mask_filled = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask_cleaned = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_cleaned = cv2.connectedComponentsWithStats(mask_cleaned, 8)[1] > 0  # Clean tiny blobs

    clean_masks.append(mask_cleaned)
    areas.append(np.sum(mask_cleaned))

# === SHOW RESULTS WITH OVERLAY ===

for i in range(num_images):
    img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    # original
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original')
    plt.axis('off')

    # overlay
    plt.subplot(1, 2, 2)
    overlay = img_rgb.copy()
    overlay[clean_masks[i]] = [0, 0, 255]
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    plt.imshow(blended)
    plt.title(f'Overlayed — Area: {areas[i]} px')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# === TABLE OF RESULTS ===

df = pd.DataFrame({
    'Filename': image_files,
    'RedAreaPixels': areas
})
print(df)

# === BAR CHART ===

plt.figure(figsize=(10, 6))
plt.bar(df['Filename'], df['RedAreaPixels'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Red Area [pixels]')
plt.xlabel('Image Filename')
plt.title('Red-Filled Circle Area per Image')
plt.grid(True)
plt.tight_layout()
plt.show()

# === LINE PLOT ===

plt.figure(figsize=(10, 6))
plt.plot(df['RedAreaPixels'], '-o', color='dodgerblue', linewidth=2)
plt.xticks(ticks=range(num_images), labels=df['Filename'], rotation=45, ha='right')
plt.ylabel('Red Area [pixels]')
plt.xlabel('Image Filename')
plt.title('Trend of Red Area across Images')
plt.grid(True)
plt.tight_layout()
plt.show()

# === PIE CHART ===

plt.figure(figsize=(8, 8))
plt.pie(df['RedAreaPixels'], labels=df['Filename'], autopct='%1.1f%%', startangle=140)
plt.title('Proportional Red Area per Image')
plt.tight_layout()
plt.show()
