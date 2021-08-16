from pathlib import Path
import geopandas as gpd
from data import preprocess_dir
import shutil
import sys
sys.path.append("..")
from options.process_options import ProcessOptions

# read in options and labels data frame
opts = ProcessOptions().parse()
in_dir = Path(opts.in_dir)
y = gpd.read_file(opts.label_path)
out_dir = Path(opts.out_dir)

# decide how to split glacier lakes across splits
if opts.split:
    basin_mapping = {
        "train": ["Arun", "Bheri", "Budhi Gandaki", "Dudh Koshi", "Humla", "Indrawati", "Kali", "Kali Gandaki"],
        "val": ["Karnali", "Kawari", "Likhu", "Marsyangdi", "Mugu", "Seti"],
        "test": ["Sun Koshi", "Tama Koshi", "Tamor", "Tila", "Trishuli", "West Seti"]
    }
else:
    basin_mapping = {"all": y["Sub_Basin"].unique().tolist()}

# create directories for processing
for split in basin_mapping.keys():
    if (out_dir / split).exists():
        shutil.rmtree(out_dir / split)
    (out_dir / split).mkdir(parents=True)

# identify lakes that go into each split
ids = {}
for split in basin_mapping.keys():
    ids[split] = list(y[y.Sub_Basin.isin(basin_mapping[split])].GL_ID.values)

    if opts.subset_size is not None:
        ids[split] = ids[split][:opts.subset_size]

# copy lakes across splits
for path in in_dir.glob("*tif"):
    for split in ids.keys():
        for i in ids[split]:
            if path.stem.find(i) != -1:
                path.rename(out_dir / split / path.name)

# preprocess all the lakes
for split in basin_mapping.keys():
    preprocess_dir(out_dir / split, y)

if not opts.split:
    (out_dir / "all" / "images").rename(out_dir / "images")
    (out_dir / "all" / "labels").rename(out_dir / "labels")
    (out_dir / "all" / "meta").rename(out_dir / "meta")
    (out_dir / "all" / "statistics.csv").rename(out_dir / "statistics.csv")
    shutil.rmtree(out_dir / "all")
