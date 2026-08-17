import os
from datetime import datetime, timezone
from typing import List

import pandas as pd

from why_do_I_keep_losing.api.opendota import (
    get_all_pro_matches,
    get_match_details,
)
from why_do_I_keep_losing.core.match import MatchSummary

from pathlib import Path


PROCESSED_DIR = Path("data/processed")


def get_last_saved_match_id() -> int | None:
    """
    Find the oldest match ID currently stored in processed
    Parquet files.

    Returns None if no Parquet files exist.
    """

    parquet_files = list(
        PROCESSED_DIR.glob("pro_matches_*.parquet")
    )

    if not parquet_files:
        print("[*] No existing Parquet files found.")
        return None

    print(
        f"[*] Found {len(parquet_files)} existing Parquet files."
    )

    oldest_match_id = None

    for parquet_file in parquet_files:

        print(f"    Reading {parquet_file.name}")

        df = pd.read_parquet(
            parquet_file,
            columns=["match_id"],
        )

        if df.empty:
            continue

        file_oldest_id = int(df["match_id"].min())

        if (
            oldest_match_id is None
            or file_oldest_id < oldest_match_id
        ):
            oldest_match_id = file_oldest_id

    if oldest_match_id is not None:
        print(
            f"[*] Oldest saved match ID: "
            f"{oldest_match_id}"
        )

    return oldest_match_id

def export_to_parquet(
    matches: List[MatchSummary],
    filename: str | None = None,
):
    """Flatten MatchSummary objects into rows and save to a Parquet file."""

    if not matches:
        print("[!] No matches to export.")
        return

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Get the actual match ID range in this batch
    min_match_id = min(match.match_id for match in matches)
    max_match_id = max(match.match_id for match in matches)

    # Generate filename automatically
    if filename is None:
        filename = (
            f"pro_matches_{min_match_id}_{max_match_id}.parquet"
        )

    output_path = os.path.join(output_dir, filename)

    # Timestamp for this download batch
    downloaded_at = datetime.now(timezone.utc).isoformat()

    rows = []

    for match in matches:

        radiant_heroes = [
            {
                "hero_id": p.hero_id,
                "role": p.role,
            }
            for p in match.picks
            if p.team == "radiant"
        ]

        dire_heroes = [
            {
                "hero_id": p.hero_id,
                "role": p.role,
            }
            for p in match.picks
            if p.team == "dire"
        ]

        rows.append({
            "match_id": match.match_id,
            "patch": match.patch,
            "winning_team": match.winning_team,
            "radiant_heroes": radiant_heroes,
            "dire_heroes": dire_heroes,
            "downloaded_at": downloaded_at,
        })

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(
        subset=["match_id"]
    )
    df.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nSuccessfully saved {len(rows)} records"
        f"\nMatch IDs: {min_match_id} -> {max_match_id}"
        f"\nFile: {output_path}"
    )


def download_pro_matches(total_matches: int = 2000):
    print("Getting Pro matches...")
    last_saved_match_id = get_last_saved_match_id()

    if last_saved_match_id is not None:
        print(
            f"[*] Continuing from match ID "
            f"{last_saved_match_id}"
        )
    else:
        print(
            "[*] No previous data found. "
            "Starting from newest pro matches."
        )

    pro_matches = get_all_pro_matches(
        target_match_count=total_matches,
        less_than_match_id=last_saved_match_id
    )

    print(f"Found: {len(pro_matches)} new match headers")

    match_objects: List[MatchSummary] = []

    try:
        for idx, raw_summary in enumerate(pro_matches):

            match_id = raw_summary["match_id"]

            print(
                f"[{idx + 1}/{len(pro_matches)}] "
                f"Fetching details for match {match_id}..."
            )

            raw_details = get_match_details(match_id)

            if not raw_details:
                print(f"[!] No details found for match {match_id}")
                continue

            match_obj = MatchSummary.from_dict(raw_details)

            match_objects.append(match_obj)

    except KeyboardInterrupt:
        print("\n[!] Script manually interrupted with Ctrl+C")

    finally:
        print("\n[*] Flushing remaining buffer to disk...")

        export_to_parquet(match_objects)


if __name__ == "__main__":
    download_pro_matches()
