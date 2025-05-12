# Mount Google Drive (only needed in Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.stats import variation, skew, kurtosis, entropy
from scipy.spatial.distance import pdist

# === NOTE ===
# Images are processed in blocks (e.g., 5 at a time) due to high file size and memory usage in Colab.
# Adjust the 'block' range manually as needed to iterate through the full dataset.

# --- Metrics functions ---

def gini_coefficient(x):
    """Calculate the Gini coefficient for inequality."""
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n

def circularity(region):
    """Calculate circularity of a labeled region."""
    if region.perimeter == 0:
        return 0
    return 4 * np.pi * region.area / (region.perimeter ** 2)

def fractal_dimension(Z):
    """Estimate fractal dimension using box-counting method."""
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])
    Z = (Z < 0.5)
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def multifractal_dimensions(gray_array, threshold=0.1):
    """Calculate D1 and D2 multifractal dimensions."""
    img = gray_array.copy()
    img[img < threshold] = 0.0
    box_sizes = np.array([2, 4, 8, 16, 32, 64])
    d1_vals, d2_vals = [], []

    for size in box_sizes:
        n_rows = img.shape[0] // size
        n_cols = img.shape[1] // size
        p_vals = []
        for i in range(n_rows):
            for j in range(n_cols):
                box = img[i*size:(i+1)*size, j*size:(j+1)*size]
                total = np.sum(box)
                if total > 0:
                    p_vals.append(total)
        p_vals = np.array(p_vals)
        p_vals /= np.sum(p_vals)
        d1 = -np.sum(p_vals * np.log(p_vals))
        d2 = np.sum(p_vals**2)
        d1_vals.append(d1)
        d2_vals.append(d2)

    epsilons = 1.0 / box_sizes
    log_eps = np.log(epsilons)
    log_d1 = np.array(d1_vals)
    log_d2 = np.log(d2_vals)
    slope_d1, _ = np.polyfit(log_eps, log_d1, 1)
    slope_d2, _ = np.polyfit(log_eps, log_d2, 1)
    return slope_d1, -slope_d2

# --- Main image analysis ---

def analyze_image_full(image_path, output_dir):
    """Analyze one image and return a dictionary of metrics."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(image_rgb)

    thresh_val = threshold_otsu(gray)
    binary = gray > thresh_val
    label_img = label(binary)

    # Structural metrics
    total_pixels = binary.size
    coverage = (np.sum(binary) / total_pixels) * 100
    num_fragments = np.max(label_img)
    fractal_dim = fractal_dimension(binary)

    # Texture & statistical metrics
    flat_gray = gray.flatten()
    std_intensity = np.std(flat_gray)
    coef_var = variation(flat_gray)
    gini_index = gini_coefficient(flat_gray)
    skewness = skew(flat_gray)
    kurt = kurtosis(flat_gray)
    hist, _ = np.histogram(flat_gray, bins=256, range=(0,1), density=False)
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    ent = entropy(hist, base=2)

    grad_y, grad_x = np.gradient(gray)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_mean = np.mean(gradient_magnitude)

    # Shape metrics
    props = regionprops(label_img)
    circularities = [circularity(p) for p in props if p.area > 0]
    mean_circularity = np.mean(circularities) if circularities else 0

    centroids = [p.centroid for p in props]
    mean_distance = np.mean(pdist(centroids)) if len(centroids) > 1 else 0

    # Multifractal metrics
    D1, D2 = multifractal_dimensions(gray)

    # Save panel
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original image")
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title("Binary mask")
    axes[2].imshow(label_img, cmap='nipy_spectral')
    axes[2].set_title("Labeled regions")
    heatmap = 1 - gradient_magnitude / np.max(gradient_magnitude)
    axes[3].imshow(heatmap, cmap='plasma')
    axes[3].set_title("Gradient heatmap")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_panel.tiff"), format='tiff')
    plt.close()

    return {
        "ID": base_name,
        "Coverage (%)": coverage,
        "Num. fragments": num_fragments,
        "Fractal dim.": fractal_dim,
        "Std. intensity": std_intensity,
        "Coeff. variation": coef_var,
        "Gini index": gini_index,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "Entropy": ent,
        "Gradient mean": gradient_mean,
        "Mean circularity": mean_circularity,
        "Centroid distance": mean_distance,
        "D1 (info dim.)": D1,
        "D2 (corr. dim.)": D2
    }

# --- Execution block (adjust paths as needed) ---

input_folder = '/content/drive/MyDrive/your_project/images_tiff'
output_folder = '/content/drive/MyDrive/your_project/output_metrics'
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))])

# Process a block of images
block = image_files[0:2]  # Change range as needed
results = []

for filename in block:
    path = os.path.join(input_folder, filename)
    metrics = analyze_image_full(path, output_folder)
    results.append(metrics)

# Save CSV
df = pd.DataFrame(results)
csv_name = "metrics_block0.csv"
df.to_csv(os.path.join(output_folder, csv_name), index=False)

print("✅ Block processed and saved:", csv_name)
