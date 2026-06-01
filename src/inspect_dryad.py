"""
inspect_dryad.py
================
Validates your extracted Dryad dataset and shows a summary.
Run this BEFORE generating masks to confirm everything is in place.

Usage:
    python src/inspect_dryad.py \
        --images_dir data/dryad/images_original \
        --tables_dir data/dryad/Tables
"""

import re, argparse
from pathlib import Path

try:
    from docx import Document
except ImportError:
    raise ImportError("Run:  pip install python-docx")


def main(images_dir, tables_dir):
    images_path = Path(images_dir)
    tables_path = Path(tables_dir)

    img_ids = {}
    for f in images_path.iterdir():
        m = re.match(r'^([\d\.]+)\s*', f.name)
        if m:
            img_id = m.group(1).rstrip('.').strip()
            img_ids[img_id] = f

    tbl_ids = {}
    for f in tables_path.glob("*.docx"):
        m = re.search(r'Table\s+([\d\.]+)', f.name)
        if m:
            tbl_id = m.group(1).rstrip('.').strip()
            tbl_ids[tbl_id] = f

    matched = set(img_ids) & set(tbl_ids)

    print(f"\n{'='*50}")
    print(f"  Images found  : {len(img_ids)}")
    print(f"  Tables found  : {len(tbl_ids)}")
    print(f"  Matched pairs : {len(matched)}")
    print(f"{'='*50}")

    # preview one table
    if tbl_ids:
        sample_id  = sorted(tbl_ids)[0]
        sample_doc = Document(str(tbl_ids[sample_id]))
        print(f"\nSample table ({sample_id}):")
        for row in sample_doc.tables[0].rows:
            print("  ", [c.text.strip() for c in row.cells])

    if len(matched) == 0:
        print("\n❌  No pairs found! Check your folder paths.")
    elif len(matched) < len(img_ids):
        print(f"\n⚠️  {len(img_ids)-len(matched)} images have no matching table.")
    else:
        print("\n✅  All images have matching tables. Ready to generate masks.")

    print(f"\nNext step:")
    print(f"  python src/generate_masks_dryad.py \\")
    print(f"      --images_dir {images_dir} \\")
    print(f"      --tables_dir {tables_dir} \\")
    print(f"      --masks_dir  data/dryad/masks \\")
    print(f"      --visualise")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--tables_dir", required=True)
    args = parser.parse_args()
    main(args.images_dir, args.tables_dir)
