import os
import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from PIL import Image
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
# from transformers.models.llama.modeling_llama import LlamaAttention
from .hf_utils import llama_attn_forward, llama_new_forward


def llama_modify(model, start_layer, end_layer, use_attn, alpha, use_cfg,
                 img_start_idx, img_end_idx):
    layers = model.language_model.model.layers
    for i in range(start_layer, end_layer):
        layers[i].self_attn.use_attn = use_attn
        layers[i].self_attn.alpha = alpha
        layers[i].self_attn.use_cfg = use_cfg
        layers[i].self_attn.img_start_idx = img_start_idx
        layers[i].self_attn.img_end_idx = img_end_idx
        layers[i].self_attn.forward = types.MethodType(llama_new_forward, layers[i].self_attn)

def llama_reset(model, start_layer, end_layer):
    layers = model.language_model.model.layers
    for i in range(start_layer, end_layer):
        # layers[i].self_attn.forward = types.MethodType(llama_forward, layers[i].self_attn)
        layers[i].self_attn.forward = types.MethodType(llama_attn_forward, layers[i].self_attn)


def generate_qa_input(question, label, raw_image, processor, device="cuda"):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors='pt',
        padding=False
    )

    target_ids = processor.tokenizer(label, return_tensors='pt', add_special_tokens=False)
    inputs["input_ids"] = inputs.input_ids = torch.cat(
        (inputs.input_ids, target_ids.input_ids), dim=1
    )
    inputs["attention_mask"] = inputs.attention_mask = torch.cat(
        (inputs.attention_mask, target_ids.attention_mask), dim=1
    )
    inputs.labels = inputs.input_ids.clone()
    inputs.labels[0, :-len(target_ids.input_ids[0])] = -100
    inputs["labels"] = inputs.labels

    return inputs.to(device, torch.float16)


def generate_completion_input(question, label, raw_image, processor, device="cuda"):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    prompt += " This is a picture of"

    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors='pt',
        padding=False
    )

    target_ids = processor.tokenizer(label, return_tensors='pt', add_special_tokens=False)
    inputs["input_ids"] = inputs.input_ids = torch.cat(
        (inputs.input_ids, target_ids.input_ids), dim=1
    )
    inputs["attention_mask"] = inputs.attention_mask = torch.cat(
        (inputs.attention_mask, target_ids.attention_mask), dim=1
    )
    inputs.labels = inputs.input_ids.clone()
    inputs.labels[0, :-len(target_ids.input_ids[0])] = -100
    inputs["labels"] = inputs.labels

    return inputs.to(device, torch.float16)


def visualize(image, mask, img_name):
    colors = [(1, 1, 1, 0), (1, 0, 0, 1)]  # RGBA: white-transparent to solid red
    custom_cmap = LinearSegmentedColormap.from_list("transparent_to_red", colors)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Display the image
    ax.imshow(image, interpolation="nearest", extent=(0, 336, 0, 336))

    # Overlay the mask as a transparent layer, resizing it to 14x14 patches
    mask_resized = np.kron(mask, np.ones((14, 14)))  # Each value expands to a 14x14 block
    ax.imshow(mask_resized, cmap=custom_cmap, alpha=0.7, extent=(0, 336, 0, 336))

    # Add a colorbar for the mask
    cbar = plt.colorbar(ax.imshow(mask_resized, cmap=custom_cmap, alpha=0.7, extent=(0, 336, 0, 336)))
    cbar.set_label("Mask Intensity")
     # Remove axes for cleaner visualization
    ax.axis("off")

    # Display the visualization
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./figures/{img_name}.jpg")


def bbox_to_mask(bbox, grid_size=24):
    """
    Transforms a normalized bounding box (proportional coordinates) into a 24x24 binary mask.

    Args:
        bbox (tuple): Bounding box as (x_min, y_min, width, height), values in [0, 1].
        grid_size (int): Number of patches along one dimension (default is 24x24).

    Returns:
        np.ndarray: Binary mask of shape (grid_size, grid_size).
    """
    # Initialize the binary mask
    mask = np.zeros((grid_size, grid_size), dtype=int)

    # Extract normalized bounding box coordinates
    x_min, y_min, box_width, box_height = bbox
    x_max = x_min + box_width
    y_max = y_min + box_height

    # Iterate through each patch
    for i in range(grid_size):
        for j in range(grid_size):
            # Compute patch boundaries in normalized coordinates
            patch_x_min = j / grid_size
            patch_y_min = i / grid_size
            patch_x_max = (j + 1) / grid_size
            patch_y_max = (i + 1) / grid_size

            # Compute overlap area between the patch and bounding box
            overlap_x_min = max(x_min, patch_x_min)
            overlap_y_min = max(y_min, patch_y_min)
            overlap_x_max = min(x_max, patch_x_max)
            overlap_y_max = min(y_max, patch_y_max)

            # Calculate the overlap area
            overlap_width = max(0, overlap_x_max - overlap_x_min)
            overlap_height = max(0, overlap_y_max - overlap_y_min)
            overlap_area = overlap_width * overlap_height

            # Check if there is sufficient overlap
            patch_area = 1 / (grid_size ** 2)  # Each patch area in normalized coordinates
            if overlap_area > 0.5 * patch_area:  # Threshold: at least 50% overlap
                mask[i, j] = 1

    return mask


def evaluate_precision_recall(predicted_mask, ground_truth_mask, threshold=0.5):
    """
    Calculates precision and recall for a segmentation task.

    Args:
        predicted_mask (np.ndarray): Predicted mask of shape (H, W) or (N, H, W), values in [0, 1].
        ground_truth_mask (np.ndarray): Ground truth binary mask of shape (H, W) or (N, H, W), values in {0, 1}.
        threshold (float): Threshold to binarize the predicted mask. Defaults to 0.5.

    Returns:
        dict: Dictionary containing precision and recall.
    """
    # Binarize the predicted mask
    predicted_binary = (predicted_mask >= threshold).astype(int)
    
    # Ensure the ground truth mask is binary
    ground_truth_binary = (ground_truth_mask > 0).astype(int)
    
    # Flatten the arrays for easier computation
    predicted_flat = predicted_binary.flatten()
    ground_truth_flat = ground_truth_binary.flatten()
    
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = np.sum((predicted_flat == 1) & (ground_truth_flat == 1))
    fp = np.sum((predicted_flat == 1) & (ground_truth_flat == 0))
    fn = np.sum((predicted_flat == 0) & (ground_truth_flat == 1))
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {"precision": precision, "recall": recall}