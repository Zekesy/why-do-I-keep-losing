""" 
Script to generate JSON files containing wins and losses for 
each hero split up by patch.

Parquet files is assumed to be in HeroPicViT dataset. 
"""

from collections import defaultdict
import pandas as pd 
import json

from pathlib import Path

PARQUET_DIR = Path("data/processed/")
STATS_DIR = Path("data/stats/")

parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))


if not parquet_files:
    raise FileNotFoundError(
        f"No parquet files found in {parquet_dir}"
    )

dfs = [
    pd.read_parquet(parquet_file)
    for parquet_file in parquet_files
]



df = pd.concat(dfs, ignore_index=True)


for patch, patch_df in df.groupby("patch"):
    hero_stats = defaultdict(lambda: {
        "games": 0,
        "wins": 0,
    })

    for _, row in df.iterrows():
        radiant_heroes = row["radiant_heroes"]
        dire_heroes = row["dire_heroes"]
        winning_team = row["winning_team"]

        for hero in radiant_heroes:
            hero_id = hero["hero_id"]
            hero_stats[str(hero_id)]["games"] += 1
            if winning_team == "radiant":
                hero_stats[str(hero_id)]["wins"] += 1
        
        for hero in dire_heroes:
            hero_id = hero["hero_id"]
            hero_stats[str(hero_id)]["games"] += 1
            if winning_team == "dire":
                hero_stats[str(hero_id)]["wins"] += 1
    output = {"patch": patch,
              "heroes": dict(hero_stats)} 

    output_file = STATS_DIR / f"patch_{patch}.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved {output_file}")

