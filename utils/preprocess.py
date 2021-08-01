from pathlib import Path
import geopandas as gpd
from data import preprocess_dir
import shutil
import sys
sys.path.append("..")
from options.process_options import ProcessOptions

opts = ProcessOptions().parse()
basin_mapping = {
    "train": ["Arun", "Bheri", "Budhi Gandaki", "Dudh Koshi", "Humla", "Indrawati", "Kali", "Kali Gandaki"],
    "val": ["Karnali", "Kawari", "Likhu", "Marsyangdi", "Mugu", "Seti"],
    "test": ["Sun Koshi", "Tama Koshi", "Tamor", "Tila", "Trishuli", "West Seti"]
}

in_dir = Path(opts.in_dir)
y = gpd.read_file(opts.label_path)
out_dir = Path(opts.out_dir)

for split in basin_mapping.keys():
    if (out_dir / split).exists():
        shutil.rmtree(out_dir / split)
    (out_dir / split).mkdir(parents=True)

ids = {}
for split in basin_mapping.keys():
    ids[split] = list(y[y.Sub_Basin.isin(basin_mapping[split])].GL_ID.values)

    if opts.subset_size is not None:
        ids[split] = ids[split][:opts.subset_size]

for path in in_dir.glob("*tif"):
    for split in ids.keys():
        for i in ids[split]:
            if path.stem.find(i) != -1:
                shutil.copy2(path, out_dir / split)

for split in basin_mapping.keys():
    preprocess_dir(out_dir / split, y)
