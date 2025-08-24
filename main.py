import numpy as np
import cv2
import matplotlib.pyplot as plt
import math 
from PIL import Image
import imagehash

def load_image(img_path):
    """
    Loads grayscale image from a given path
    """
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    return img

def binarize_image(img):
    """
    Binarizes a grayscale image using Otsu's thresholding.
    Returns a binary image (0 = black, 255 = white).
    """
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)

def get_bounding_boxes(binary_img, min_area=30):
    """
    Finds bounding boxes of connected components in a binary image.
    Filters out components smaller than min_area.
    Returns:
        List of bounding boxes: [ (x, y, w, h), ... ]
    """
    # Find connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_img)

    bounding_boxes = []
    
    for i in range(1, num_labels):  # start from 1 to skip the background
        x, y, w, h, area = stats[i]
        
        if area >= min_area:
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

def extract_patches(img, bboxes, padding):
    """
    Extract patches from the original grayscale image based on bounding boxes.
    
    Parameters:
    - img: Original grayscale image (uint8).
    - bboxes: List of (x, y, w, h) bounding boxes.
    - padding: Extra pixels to pad around each box.
    
    Returns:
    - List of patches (as small grayscale images, numpy arrays).
    """

    patches = []

    for (x, y, w, h) in bboxes:
        # Add padding, but clip to image size
        x1 = x - padding
        y1 = y - padding
        x2 = x + w + padding
        y2 = y + h + padding

        patch = img[y1:y2, x1:x2]
        patches.append(patch)

    return patches

def hash_patches(patches, hash_size=8):
    """
    Computes the average hash (aHash) for each patch.
    
    Parameters:
    - patches: list of grayscale patches (numpy arrays).
    - hash_size: size of the hash (default 8x8).
    
    Returns:
    - List of hashes (imagehash.ImageHash objects).
    """

    hashes = []
    
    for patch in patches:
        # Convert to PIL Image
        pil_patch = Image.fromarray(patch)
        
        # Resize to (hash_size x hash_size) inside average_hash if needed
        h = imagehash.average_hash(pil_patch, hash_size=hash_size)
        
        hashes.append(h)

    return hashes


def group_similar_patches(hashes, max_distance=0):
    """
    Groups patches whose hashes are identical (distance <= max_distance).
    
    Parameters:
    - hashes: List of imagehash.ImageHash objects.
    - max_distance: Maximum Hamming distance to consider as identical (default 0 = exact match).
    
    Returns:
    - groups: List of groups, each group is a list of patch indices.
    """

    groups = []
    used = set()

    for i in range(len(hashes)):
        if i in used:
            continue
        
        group = [i]
        used.add(i)
        
        for j in range(i+1, len(hashes)):
            if j in used:
                continue
            
            if hashes[i] - hashes[j] <= max_distance:
                group.append(j)
                used.add(j)
        
        groups.append(group)

    return groups

def match_patches_group(patches, group, threshold):
    """
    Performs comparisons inside a single patch group to find matches.
    """

    matches = []

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            patch_i = patches[group[i]]
            patch_j = patches[group[j]]

            if patch_i.shape[0] != patch_j.shape[0]:
            	continue

            if patch_i.shape[1] != patch_j.shape[1]:
                continue 

            total_pixels = patch_i.shape[0] * patch_i.shape[1]
            num_equal = np.sum(patch_i == patch_j)
            similarity = num_equal / total_pixels
            if similarity >= threshold:
                matches.append((group[i], group[j], similarity))

    return matches

def match_patches(patches, patch_groups, threshold):
    """
    Applies all vs all template matching to each group of patches and prints results.
    
    Parameters:
    - patches: List of grayscale patches (numpy arrays).
    - patch_groups: List of patch groups (each group is a list of indices).
    - threshold: Minimum similarity for a match (default 0.9).
    
    Returns:
    - None
    """
    all_matches = []
    
    for idx, group in enumerate(patch_groups):
        #print(f"Processing group {idx}...")
        
        matches = match_patches_group(patches, group, threshold)
        
        if matches:
            #print(f"Matches found in group {idx}: {matches}")
            all_matches.extend(matches)
    
    return all_matches

def visualize_matches_on_document(img, patch_bboxes, matches, output_dir):
    """
    Visualizes matching patches on the document image.
    """
    # Convert grayscale to color
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    report_lines = []

    if not matches:
        report_lines.append("No detection")
    else:
        report_lines.append("Detections:")
    

    for match in matches:    
        idx1, idx2, sim = match


        x1, y1, w1, h1 = patch_bboxes[idx1]
        x2, y2, w2, h2 = patch_bboxes[idx2]

        # Draw rectangles
        cv2.rectangle(img_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.rectangle(img_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

        # Draw line between centers
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        cv2.line(img_color, center1, center2, (255, 0, 0), 2)

        line = f"{center1} - {center2} : {sim:.4f}"
        report_lines.append(line)

    # Save image
    cv2.imwrite(os.path.join(output_dir, "detection.png"), img_color)

    # Save report
    with open(os.path.join(output_dir, "detection.txt"), "w") as f:
        for line in report_lines:
            f.write(line + "\n")

from pathlib import Path
import os

def detect(file, threshold = 0.8, hash_size = 8, separate_results=False):
    max_distance = np.floor((1-threshold)*hash_size*hash_size)
    img = load_image(file)
    binary = binarize_image(img)
    boxes = get_bounding_boxes(binary)
    patches = extract_patches(img, boxes, padding = -1)
    hashes = hash_patches(patches, hash_size)
    groups = group_similar_patches(hashes, max_distance)
    matches = match_patches(patches, groups, threshold)
    if separate_results:
        img_name = Path(file).stem
        output_dir = os.path.join("results", img_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "."  
    visualize_matches_on_document(img, boxes, matches, output_dir)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("-t")
    parser.add_argument("--separate-results", action="store_true")
    parser.parse_args()
    args = parser.parse_args()
    file = args.image
    threshold = float(args.t)
    detect(file, threshold, hash_size = 8, separate_results = args.separate_results)


