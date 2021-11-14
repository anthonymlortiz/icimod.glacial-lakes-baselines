from pathlib import Path
from options.infer_options import PathOptions
import utils.data as dt

# read in options and labels data frame
opts = PathOptions().parse()

# create paths for the validation sets
image_dir = Path(opts.image_dir)
inference_dir = Path(opts.inference_dir)
output_dir = Path(opts.output_dir)
labeling_dir = Path(opts.labeling_dir)
results = {}

for split_type in ["test", "val"]:
    split_dir = image_dir / f"splits/{split_type}/images"
    ids = dt.list_ids(split_dir)
    output_name = f"{opts.dataset}_{split_type}-{opts.model}"
    results[output_name] = dt.eval_paths(ids["sample_id"], inference_dir)

ids = dt.list_ids(image_dir / "images")
results[f"{opts.dataset}-{opts.model}"] = dt.eval_paths(ids["sample_id"], inference_dir)
ids = dt.list_ids(labeling_dir)
results[f"{opts.dataset}-{opts.model}_recent"] = dt.eval_paths(ids["sample_id"], inference_dir)

for k, v in results.items():
    v.to_csv(output_dir / f"{k}.csv", index=False)
