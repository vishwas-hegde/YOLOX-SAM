import os
import json
from PIL import Image

# Paths to the folders containing images and annotations
images_folder = "path/to/images_folder"
annotations_folder = "path/to/annotations_folder"
coco_output_file = "output_coco_annotations.json"

# Initialize COCO format structure
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "object", "supercategory": "none"}]
}

# Helper variables
annotation_id = 1  # COCO annotation IDs must be unique
def get_image_info(image_id, file_name, width, height):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

def get_annotation_info(annotation_id, image_id, category_id, bbox):
    x, y, w, h = bbox
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0
    }

def parse_bbox_line(line):
    """Parses a single line from a .box file and returns the bounding box in COCO format."""
    _, coords = line.split(": ")
    x1, y1, x2, y2 = map(int, coords.split())
    return x1, y1, x2 - x1, y2 - y1

# Process each image and corresponding annotation
for image_id, image_file in enumerate(os.listdir(images_folder), start=1):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Get image dimensions
    image_path = os.path.join(images_folder, image_file)
    with Image.open(image_path) as img:
        width, height = img.size

    # Add image information to COCO data
    coco_data["images"].append(get_image_info(image_id, image_file, width, height))

    # Process corresponding .box file
    annotation_file = os.path.join(annotations_folder, os.path.splitext(image_file)[0] + ".box")
    if not os.path.exists(annotation_file):
        print(f"Warning: No annotation file for image {image_file}")
        continue

    with open(annotation_file, "r") as f:
        for line in f:
            bbox = parse_bbox_line(line.strip())
            coco_data["annotations"].append(
                get_annotation_info(annotation_id, image_id, 1, bbox)
            )
            annotation_id += 1

# Save COCO annotations to a JSON file
with open(coco_output_file, "w") as json_file:
    json.dump(coco_data, json_file, indent=4)

print(f"COCO annotations saved to {coco_output_file}")
