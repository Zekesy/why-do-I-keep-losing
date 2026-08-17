from pathlib import Path

import time
from typing import List, Dict, Any

import requests
import json
import re


OPENDOTA_API = "https://api.opendota.com/api"


def get_opendota_heroes() -> List[dict]:
    """Get canonical hero data"""
    response = requests.get(
        f"{OPENDOTA_API}/heroes",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_pro_matches() -> List[dict]:
    """Get pro matches, gets latest 100 only"""
    response = requests.get(
        f"{OPENDOTA_API}/proMatches",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()

def get_all_pro_matches(
    target_match_count: int = 2000,
    less_than_match_id: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Paginate through /proMatches.

    If less_than_match_id is provided, only return matches
    older than that match ID.

    OpenDota returns pro matches newest -> oldest.
    """

    all_matches = []
    last_match_id = less_than_match_id

    while len(all_matches) < target_match_count:
        params = {}

        if last_match_id is not None:
            params["less_than_match_id"] = last_match_id

        try:
            response = requests.get(
                f"{OPENDOTA_API}/proMatches",
                params=params,
                timeout=10,
            )

            response.raise_for_status()

            batch = response.json()

        except requests.exceptions.RequestException as e:
            print(
                f"[!] Error fetching match page: {e}. "
                "Stopping pagination."
            )
            break

        if not batch:
            print("[!] No more pro matches available.")
            break

        all_matches.extend(batch)

        # The final item is the oldest match in this page.
        last_match_id = batch[-1]["match_id"]

        print(
            f"Fetched "
            f"{len(all_matches)}/{target_match_count} "
            f"match headers... "
            f"(Oldest ID: {last_match_id})"
        )

        time.sleep(1.0)

    return all_matches[:target_match_count]

def get_match_details(match_id, retries: int = 3) -> dict:
    """Get a single match details"""
    url = f"{OPENDOTA_API}/matches/{match_id}"

    for attempt in range(1, retries + 1):
        try:
            time.sleep(2.0)

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                print(
                    f"Rate limited (429) on match {match_id}. Sleeping 5 seconds..."
                )
                time.sleep(5)
                continue
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            print(
                f"Attempt {attempt}/{retries} timed out for match {match_id}. Retrying..."
            )
            time.sleep(2)
    
    print(
        f"Failed to fetch match {match_id} after {retries} attempts. Skipping."
    )
    return {}
