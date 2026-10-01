"""Download an embedding model into a plain directory, for the Docker image.

The Hugging Face cache stores files as symlinks into separate blob directories.
onnxruntime >= 1.30 refuses to load external weights (model.onnx_data) that resolve
outside the model's directory, so the image ships a dereferenced copy instead.

Run: python scripts/fetch_embedding_model.py <model name> <target dir>
"""

import shutil
import sys
import tempfile
from pathlib import Path

from fastembed import TextEmbedding


def main() -> None:
    name, target = sys.argv[1], Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as cache:
        downloaded = TextEmbedding(name, cache_dir=cache, lazy_load=True)
        shutil.copytree(downloaded.model._model_dir, target)  # follows symlinks
    # Load from the copy once, so a broken image fails the build, not the first request.
    next(iter(TextEmbedding(name, specific_model_path=str(target)).embed(["warm-up"])))
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 2**20
    print(f"{name} -> {target} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
