# scripts/download_data.py
from pathlib import Path
import shutil
import sys
import kagglehub

def main():
    # Resolve absolute project root (parent of scripts/)
    project_root = Path(__file__).resolve().parents[1]
    dst_dir = (project_root / "dataset").resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Download / locate Kaggle cache
    src_root = Path(kagglehub.dataset_download("jacopoferretti/bbc-articles-dataset")).resolve()
    print("Path to dataset files:", src_root)

    # Find all CSV files recursively
    csvs = sorted(src_root.rglob("*.csv"))
    print(f"Found {len(csvs)} CSV file(s).")

    # If none found, show a quick tree and exit
    if not csvs:
        print("No CSVs found. Contents of source root:")
        for p in sorted(src_root.iterdir()):
            print("  ", ("[DIR]" if p.is_dir() else "[FILE]"), p.name)
        sys.exit(1)

    # Copy (flatten) into dataset/
    for f in csvs:
        target = dst_dir / f.name
        shutil.copy2(f, target)
        print(f"  Copied: {f.relative_to(src_root)}  ->  {target}")

    print("\nDONE. Files are in:", dst_dir)

if __name__ == "__main__":
    main()
