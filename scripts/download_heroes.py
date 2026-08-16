from pathlib import Path
import json
import re
import time
import requests



OPENDOTA_API = "https://api.opendota.com/api"
LIQUIPEDIA_API = "https://liquipedia.net/commons/api.php"

OUTPUT_DIR = Path("data/heroes")
ICON_DIR = OUTPUT_DIR / "icons"

HEADERS = {
    "User-Agent": (
        "why-do-i-keep-losing/0.1 "
        "(Dota 2 ML research project)"
    ),
    "Accept-Encoding": "gzip",
}


def slugify(name: str) -> str:
    """Convert hero name into a filename."""

    name = name.lower()
    name = name.replace("'", "")
    name = name.replace("-", "_")
    name = re.sub(r"\s+", "_", name)

    return name


def get_opendota_heroes() -> list[dict]:
    """Get canonical hero data from OpenDota."""

    response = requests.get(
        f"{OPENDOTA_API}/heroes",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_liquipedia_hero_icons() -> list[dict]:
    """Get all hero icons from the Liquipedia Commons category."""

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Dota_2_hero_icons",
        "cmlimit": 500,
        "format": "json",
    }

    response = requests.get(
        LIQUIPEDIA_API,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["query"]["categorymembers"]


def get_liquipedia_image_url(page_id: int) -> str:
    """Get the actual image URL for a Liquipedia file."""

    params = {
        "action": "query",
        "pageids": page_id,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
    }

    response = requests.get(
        LIQUIPEDIA_API,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    page = next(iter(data["query"]["pages"].values()))

    return page["imageinfo"][0]["url"]


def download_image(url: str, path: Path) -> None:
    """Download an image."""

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    path.write_bytes(response.content)


def main():
    ICON_DIR.mkdir(parents=True, exist_ok=True)

    print("Getting heroes from OpenDota...")
    heroes = get_opendota_heroes()

    print(f"Found {len(heroes)} heroes.")
    print("Getting hero icons from Liquipedia Commons...")
    icon_files = get_liquipedia_hero_icons()

    print(f"Found {len(icon_files)} category entries.")

    icon_lookup = {}

    for file in icon_files:
        title = file["title"]

        # We only want actual files.
        if file["ns"] != 6:
            continue

        # Expected:
        # File:Anti-Mage icon dota2 gameasset.png

        prefix = "File:"
        suffix = " icon dota2 gameasset.png"

        if not title.startswith(prefix):
            continue

        if not title.endswith(suffix):
            continue

        hero_name = title[
            len(prefix):-len(suffix)
        ]

        icon_lookup[hero_name.lower()] = {
            "page_id": file["pageid"],
            "title": title,
        }

    print(f"Found {len(icon_lookup)} hero icon files.")

    metadata = {}

    for index, hero in enumerate(heroes, start=1):

        hero_id = hero["id"]
        hero_name = hero["localized_name"]

        print(
            f"\n[{index}/{len(heroes)}] "
            f"{hero_id}: {hero_name}"
        )

        icon = icon_lookup.get(hero_name.lower())

        if icon is None:

            print("  WARNING: icon not found")

            metadata[str(hero_id)] = {
                "hero_id": hero_id,
                "internal_name": hero["name"],
                "localized_name": hero_name,
                "primary_attr": hero["primary_attr"],
                "attack_type": hero["attack_type"],
                "roles": hero["roles"],
                "legs": hero["legs"],
                "icon_path": None,
                "icon_source": None,
                "source_url": None,
            }

            continue

        page_id = icon["page_id"]

        print(f"  File: {icon['title']}")
        print(f"  Page ID: {page_id}")

        # Get image URL
        image_url = get_liquipedia_image_url(page_id)

        # Download
        filename = f"{slugify(hero_name)}.png"
        output_path = ICON_DIR / filename

        if output_path.exists():
            print("  Already downloaded")
        else:
            print(f"  Downloading -> {output_path}")
            try:
                download_image(
                    image_url,
                    output_path,
                )
            except requests.RequestException as e:
                print(f"  ERROR: {e}")
                output_path = None

        # Metadata
        metadata[str(hero_id)] = {
            "hero_id": hero_id,
            "internal_name": hero["name"],
            "localized_name": hero_name,
            "primary_attr": hero["primary_attr"],
            "attack_type": hero["attack_type"],
            "roles": hero["roles"],
            "legs": hero["legs"],
            "icon_path": (
                str(output_path)
                if output_path is not None
                else None
            ),
            "icon_source": "liquipedia_commons",
            "source_url": image_url,
            "liquipedia_page_id": page_id,
        }

        # Be polite to Liquipedia.
        time.sleep(0.25)

    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # Summary
    successful = sum(
        hero["icon_path"] is not None
        for hero in metadata.values()
    )

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)

    print(f"OpenDota heroes:    {len(heroes)}")
    print(f"Liquipedia icons:   {len(icon_lookup)}")
    print(f"Downloaded icons:   {successful}")
    print(f"Missing icons:      {len(heroes) - successful}")
    print(f"Icon directory:     {ICON_DIR}")
    print(f"Metadata:           {metadata_path}")

if __name__ == "__main__": 
    main()
