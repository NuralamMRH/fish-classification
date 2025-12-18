import os
import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str)
    parser.add_argument("labels_json_path", type=str)
    parser.add_argument("out_json_path", type=str)
    parser.add_argument("--min_threshold", type=float, default=0.03)
    parser.add_argument("--max_threshold", type=float, default=0.4)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    # Convert numpy arrays to lists for JSON
    db = data["data_base"].tolist()
    internal_ids = data["internal_ids"].tolist()
    image_ids = data["image_ids"].tolist()
    annotation_ids = data["annotation_ids"].tolist()
    drawn_fish_ids = data["drawn_fish_ids"].tolist()

    with open(args.labels_json_path, "r") as f:
        labels = json.load(f)

    out = {
        "data_base": db,
        "internal_ids": internal_ids,
        "image_ids": image_ids,
        "annotation_ids": annotation_ids,
        "drawn_fish_ids": drawn_fish_ids,
        "labels": labels,
        "min_threshold": args.min_threshold,
        "max_threshold": args.max_threshold,
    }

    os.makedirs(os.path.dirname(args.out_json_path) or ".", exist_ok=True)
    with open(args.out_json_path, "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    main()
