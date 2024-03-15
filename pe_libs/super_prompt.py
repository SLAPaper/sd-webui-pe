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

"""SuperPrompt v1 from https://huggingface.co/roborovski/superprompt-v1"""


import functools as ft
import logging
import pathlib
import typing as tg

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

import modules.options as options
import modules.shared as shared

from .utils import model_path, set_seed

superprompt_path: pathlib.Path = model_path / "superprompt-v1"

enable_superprompt: bool = superprompt_path.exists()

if not enable_superprompt:
    logging.warning(
        f"SuperPrompt v1 is not available. "
        "Please clone https://huggingface.co/roborovski/superprompt-v1 into {superprompt_path}"
    )

tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
    superprompt_path, local_files_only=True
)
model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
    superprompt_path, local_files_only=True
)


@ft.lru_cache(maxsize=1024)
def super_prompt(text: str, seed: int) -> str:
    """SuperPrompt v1 from https://huggingface.co/roborovski/superprompt-v1"""
    if not enable_superprompt:
        return ""

    set_seed(seed)

    opts = tg.cast(options.Options, shared.options)
    num_tokens = getattr(opts, "SuperPrompt_V1_Max_Tokens", 150)

    with torch.inference_mode():
        input_text = f"Expand the following prompt to add more detail: {text}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

        model.to("cuda")
        outputs = model.generate(input_ids, max_length=num_tokens)
        model.to("cpu")

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
