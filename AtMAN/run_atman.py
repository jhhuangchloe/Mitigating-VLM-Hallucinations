import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from src.data import OpenImagesDataset
from src.utils import *

from tqdm import tqdm
import json


os.environ["FIFTYONE_DATABASE_DIR"] = "./data"
os.environ["FIFTYONE_DISABLE_TELEMETRY"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

dataset = OpenImagesDataset(
    # transform=partial(processor.image_processor, return_tensors="pt"),
    transform=None
)

IMAGE_TOKEN_LENGTH = 576
IMAGE_TOKEN_INDEX = 32000
IMAGE_TOKEN = "<image>"
start_layer = 0
end_layer = 31
use_attn = True
alpha = -0.9  # -f in the original paper
use_cfg = False

all_results = []

for idx, (raw_image, labels, bbox) in tqdm(
    enumerate(dataset),
    total=len(dataset)
):
    if len(labels) > 1:
        continue

    inputs = generate_completion_input(
        "What is in this picture?",
        labels[0],
        raw_image,
        processor
    )
    saliency_map = torch.zeros(IMAGE_TOKEN_LENGTH)

    with torch.no_grad():
        llama_reset(model, start_layer, end_layer)
        output = model(**inputs)
        original_loss = output.loss
        img_start_idx = torch.where(
            inputs.input_ids[0] == IMAGE_TOKEN_INDEX
        )[0][0]
        img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
        for token_start_idx in range(img_start_idx, img_end_idx):
            token_end_idx = token_start_idx + 1

            llama_modify(
                model,
                start_layer,
                end_layer,
                use_attn,
                alpha,
                use_cfg,
                token_start_idx,  # img_start_idx,
                token_end_idx, # img_end_idx,
            )

            output = model(**inputs)
            modified_loss = output.loss
            saliency_map[token_start_idx - img_start_idx] = modified_loss - original_loss

        saliency_map = saliency_map.reshape(24, 24)
        visualize(raw_image, saliency_map, f"mask{idx}")
        gt_mask = bbox_to_mask(bbox[0])
        all_results.append(evaluate_precision_recall(saliency_map.numpy(),
            gt_mask,
            threshold=torch.mean(saliency_map).item()
        ))

with open("results/atman_multithres.json", "w") as f:
    json.dump(all_results, f)
