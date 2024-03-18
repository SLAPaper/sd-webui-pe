# Copyright 2024 SLAPaper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DanTagGen-beta from https://huggingface.co/KBlueLeaf/DanTagGen-beta"""


import functools as ft
import logging
import pathlib
import typing as tg

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

import modules.options as options
import modules.shared as shared

from .utils import model_management, model_path, set_seed

dtgbeta_path: pathlib.Path = model_path / "DanTagGen-beta"

enable_dtgbeta: bool = dtgbeta_path.exists()

if not enable_dtgbeta:
    logging.warning(
        f"DanTagGen-beta is not available. "
        "Please clone https://huggingface.co/KBlueLeaf/DanTagGen-beta into {dtgbeta_path}"
    )

tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
    dtgbeta_path, local_files_only=True
)
model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
    dtgbeta_path, local_files_only=True
)


@ft.lru_cache(maxsize=1024)
def dtg_beta(
    text: str,
    seed: int,
    rating: str = "<|empty|>",
    artist: str = "<|empty|>",
    characters: str = "<|empty|>",
    copyrights: str = "<|empty|>",
    aspect_ratio: float = 0.0,
    target: str = "<|long|>",
) -> str:
    """DanTagGen-beta from https://huggingface.co/KBlueLeaf/DanTagGen-beta"""
    if not enable_dtgbeta:
        return ""

    set_seed(seed)

    input_text = f"""rating: {rating}
artist: {artist}
characters: {characters}
copyrights: {copyrights}
aspect ratio: {f"{aspect_ratio:.1f}" or '<|empty|>'}
target: {target}
general: {text}<|input_end|>"""

    with torch.inference_mode():
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
            model_management.load_device
        )
        current_token_length = int(input_ids.shape[1])
        max_token_length = current_token_length + 75 - current_token_length % 75
        max_new_tokens = max_token_length - current_token_length
        opts = tg.cast(options.Options, shared.options)
        if (
            hasattr(opts, "DanTagGen_beta_Max_New_Tokens")
            and opts.DanTagGen_beta_Max_New_Tokens is not None
            and opts.DanTagGen_beta_Max_New_Tokens > 0
        ):
            max_new_tokens = opts.DanTagGen_beta_Max_New_Tokens

        model_management.load(model)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.35,
            top_p=0.95,
            top_k=100,
        )
        model_management.offload(model)

        return tokenizer.decode(
            outputs[0][current_token_length:], skip_special_tokens=True
        )
