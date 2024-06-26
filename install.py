import logging

import launch


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


def install() -> None:
    install_deps()


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()
