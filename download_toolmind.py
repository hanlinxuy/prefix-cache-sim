import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download, HfFileSystem


def download_toolmind_dataset(output_dir: str = "data", files: list = None, force: bool = False):
    """
    Download ToolMind-Web-QA dataset files.

    Args:
        output_dir: Directory to save files
        files: List of files to download (default: all files)
        force: Force re-download even if file exists
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    repo_id = "Nanbeige/ToolMind-Web-QA"

    if files is None:
        fs = HfFileSystem()
        dataset_files = fs.ls(f"datasets/{repo_id}")
        files = [
            f["name"].split("/")[-1]
            for f in dataset_files
            if f["type"] == "file" and f["name"].endswith(".jsonl")
        ]

    print(f"ToolMind-Web-QA Dataset Downloader")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Output directory: {output_dir}")
    print(f"Files to download: {len(files)}")
    print()

    downloaded_files = []

    for i, filename in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {filename}")

        output_file = output_path / filename

        if output_file.exists() and not force:
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Already exists ({size_mb:.2f} MB) - skipping")
            downloaded_files.append(str(output_file))
            continue

        try:
            print(f"  Downloading...")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
            )

            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ Downloaded ({size_mb:.2f} MB)")

            if str(file_path) != str(output_file):
                import shutil

                shutil.copy2(file_path, output_file)
                print(f"  ✓ Copied to {output_file}")

            downloaded_files.append(str(output_file))

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

        print()

    print("=" * 60)
    print(f"Download complete!")
    print(f"Successfully downloaded: {len(downloaded_files)} files")

    total_size = sum(os.path.getsize(f) for f in downloaded_files if os.path.exists(f))
    print(f"Total size: {total_size / (1024 * 1024):.2f} MB")

    return downloaded_files


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download ToolMind-Web-QA dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data)")
    parser.add_argument(
        "--files", nargs="+", help="Specific files to download (default: all .jsonl files)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if file exists"
    )
    parser.add_argument("--list", action="store_true", help="List available files and exit")

    args = parser.parse_args()

    if args.list:
        print("Available files in ToolMind-Web-QA:")
        fs = HfFileSystem()
        dataset_files = fs.ls("datasets/Nanbeige/ToolMind-Web-QA")
        for f in dataset_files:
            if f["type"] == "file":
                size_mb = f["size"] / (1024 * 1024)
                print(f"  - {f['name'].split('/')[-1]} ({size_mb:.2f} MB)")
        return

    download_toolmind_dataset(output_dir=args.output_dir, files=args.files, force=args.force)


if __name__ == "__main__":
    main()
