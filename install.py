import logging
import pathlib
import shutil

from pe_libs.utils import curr_path, model_path

import launch

expansion_path = model_path / "expansion"

if not expansion_path.exists():
    shutil.copytree(curr_path / "expansion", expansion_path)


def install_deps() -> None:
    """install dependencies for the extension"""
    if not launch.is_installed("torch"):
        logging.warning(
            "PyTorch is not found in your environment. "
            "It's not likely the correct version if install automatically, "
            "so you have to do it manually.",
        )

    if not launch.is_installed("transformers"):
        launch.run_pip("install transformers")

    if not launch.is_installed("SentencePiece"):
        launch.run_pip("install SentencePiece")

    if not launch.is_installed("yaml"):
        launch.run_pip("install PyYAML")


def load_file_from_url(
    url: str,
    *,
    model_dir: pathlib.Path,
    progress: bool = True,
    file_name: str,
) -> None:
    """Download a file from `url` into `model_dir`, using the file present if possible."""
    model_dir.mkdir(parents=True, exist_ok=True)
    cached_file = model_dir / file_name

    if not cached_file.exists():
        logging.info(f'Downloading: "{url}" to {cached_file}')

        from torch.hub import download_url_to_file

        download_url_to_file(url, str(cached_file), progress=progress)


def download_models() -> None:
    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
        model_dir=expansion_path,
        file_name="pytorch_model.bin",
    )

    if not (model_path / "superprompt-v1").exists():
        logging.warning(
            f"You have to clone https://huggingface.co/roborovski/superprompt-v1 "
            "manually into {model_path}/superprompt-v1, or it will not work."
        )
        # not use transformers or huggingface_hub to download
        # cause it will just hang there if network is not stable


def install() -> None:
    install_deps()
    download_models()


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()
