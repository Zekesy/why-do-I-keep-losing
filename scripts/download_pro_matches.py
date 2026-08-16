from pathlib import Path
import os
import time 

from typing import List


import pandas as pd
import requests

import json
import re

from why_do_I_keep_losing.api.opendota import get_all_pro_matches, get_match_details
from why_do_I_keep_losing.core.match import MatchSummary

def export_to_parquet(matches: List[MatchSummary], filename: str = "pro_matches_heroes.parquet"):
    """Flattens MatchSummary objects into rows and saves to Parquet."""
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    
    rows = []
    for match in matches:
        radiant_heroes = [
            {"hero_id": p.hero_id, "role": p.role}
            for p in match.picks
            if p.team == "radiant"
        ]
        

        dire_heroes = [
            {"hero_id": p.hero_id, "role": p.role}
            for p in match.picks
            if p.team == "dire"
        ]

        rows.append({
            "match_id": match.match_id,
            "patch" : match.patch,
            "winning_team": match.winning_team,
            "radiant_heroes": radiant_heroes,
            "dire_heroes": dire_heroes,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Successfully saved {len(rows)} records to {output_path}")


def download_pro_matches(total_matches: int = 5000):
    print("Getting Pro matches...")
    pro_matches = get_all_pro_matches(target_match_count=total_matches) 
    print(f" Found: {len(pro_matches)} matches")

    match_objects: List[MatchSummary] = []

    for idx, raw_summary in enumerate(pro_matches):
        match_id = raw_summary["match_id"]
        print(f"[{idx + 1}/{len(pro_matches)}] Fetching details for match {match_id}...")
        raw_details = get_match_details(match_id)
        
        if not raw_details:
            continue

        match_obj = MatchSummary.from_dict(raw_details)
        match_objects.append(match_obj)

    print("\nExporting parsed dataclasses to Parquet...")
    export_to_parquet(match_objects)   

if __name__ == "__main__":
    download_pro_matches()
