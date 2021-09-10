from options.infer_options import EvalOptions
from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from tqdm import tqdm
import utils.data as dt
import utils.metrics as mt
import utils.model_utils as mu
from warnings import warn, filterwarnings
filterwarnings("ignore", category=UserWarning)

opts = EvalOptions().parse()
save_dir = Path(opts.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
eval_paths = dt.eval_paths(opts.inference_dir)

# read in the true labels, but get a buffer
vector_label = gpd.read_file(opts.vector_label)
vector_label = vector_label.set_index("GL_ID")
buffer = vector_label.buffer(distance=opts.buffer)

# loop over paths, get predictions, and evaluate
metrics = {
    "IoU": mt.IoU,
    "precision": mt.precision,
    "recall": mt.recall,
    "frechet": mt.frechet_distance
}
m = []

for i, (path, sample_id) in tqdm(eval_paths.iterrows(), total=len(eval_paths)):
    # get polygon predictions
    gl_id, _ = sample_id.split("_")
    y_hat = rasterio.open(path)
    y_hat_poly = mu.polygonize_preds(y_hat, buffer.loc[gl_id], tol=opts.tol)
    y_hat_poly.to_file(save_dir / f"{sample_id}.geojson", driver="GeoJSON")

    # get metrics for these predictions
    results = mu.polygon_metrics(
        y_hat_poly,
        vector_label.loc[gl_id:gl_id],
        y_hat,
        metrics=metrics
    )
    results["GL_ID"] = gl_id
    results["sample_id"] = sample_id
    m.append(results)

pd.DataFrame(m).to_csv(save_dir / "metrics.csv", index=False)
