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

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

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


def fill_template(
    text: str,
    rating: str,
    artist: str,
    characters: str,
    copyrights: str,
    aspect_ratio: float,
    target: str,
) -> str:
    """DanTagGen-beta prompt format"""
    return f"""
rating: {rating}
artist: {artist}
characters: {characters}
copyrights: {copyrights}
aspect ratio: {f"{aspect_ratio:.1f}" or '<|empty|>'}
target: {target}
general: {text}<|input_end|>
""".strip()


@ft.lru_cache(maxsize=1024)
def dtg_beta(
    text: str,
    seed: int,
    max_new_tokens: int,
    *,
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

    input_text = fill_template(
        text, rating, artist, characters, copyrights, aspect_ratio, target
    )

    with torch.inference_mode():
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
            model_management.load_device
        )
        if max_new_tokens <= 0:  # 填充到75*k
            current_token_length = int(input_ids.shape[1])
            max_token_length = current_token_length + 75 - current_token_length % 75
            max_new_tokens = max_token_length - current_token_length

        res_list: list[str] = []
        model_management.load(model)
        max_retry = 5
        while max_retry > 0:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.35,
                top_p=0.95,
                top_k=100,
                repetition_penalty=1.17,
                do_sample=True,
            )
            output_ids = outputs[0][current_token_length:]
            res = tokenizer.decode(output_ids, skip_special_tokens=True)

            output_token_length = len(output_ids)
            if output_token_length == 0:  # 输出为空，尝试重新生成
                max_retry -= 1
                continue
            elif output_token_length > max_new_tokens:  # 输出长度超过预定长度，放弃
                break

            # 成功产出，将输出添加到输入中，并重新计算输出长度
            res_list.append(res)
            new_text = f"{input_text}, {res}"
            input_text = fill_template(
                new_text, rating, artist, characters, copyrights, aspect_ratio, target
            )
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
                model_management.load_device
            )
            max_new_tokens -= output_token_length

        model_management.offload(model)
        return ", ".join(res_list)
