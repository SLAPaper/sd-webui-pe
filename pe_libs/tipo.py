# Copyright 2025 SLAPaper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TIPO from https://github.com/KohakuBlueleaf/z-tipo-extension
The original extension conflicted with Agent Scheduler, so reinplemented it here
Based on https://github.com/KohakuBlueleaf/KGen/blob/main/scripts/example.py
https://github.com/KohakuBlueleaf/z-tipo-extension/blob/main/scripts/tipo.py"""


import functools as ft
import logging
import pathlib
import time
import typing as tg

import kgen.executor.tipo as _tipo
import kgen.models as models
from kgen.executor.tipo import (
    apply_tipo_prompt,
    parse_tipo_request,
    parse_tipo_result,
    tipo_runner,
)
from kgen.formatter import apply_format, seperate_tags
from kgen.logging import logger

from modules.extra_networks import parse_prompt

from .prompt_utils import parse_prompt_attention
from .utils import model_path

SEED_MAX = 2**31 - 1
QUOTESWAP = str.maketrans("'\"", "\"'")
TOTAL_TAG_LENGTH = {
    "VERY_SHORT": "very short",
    "SHORT": "short",
    "LONG": "long",
    "VERY_LONG": "very long",
}
PROCESSING_TIMING = {
    "BEFORE": "Before applying other prompt processings",
    "AFTER": "After applying other prompt processings",
}
DEFAULT_FORMAT = """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>,

<|extended|>.

<|quality|>, <|meta|>, <|rating|>"""
TIMING_INFO_TEMPLATE = (
    "_Prompt upsampling will be applied to {} "
    "sd-dynamic-promps and the webui's styles feature are applied_"
)
INFOTEXT_KEY = "TIPO Parameters"
INFOTEXT_KEY_PROMPT = "TIPO prompt"
INFOTEXT_NL_PROMPT = "TIPO nl prompt"
INFOTEXT_KEY_FORMAT = "TIPO format"

PROMPT_INDICATE_HTML = """
<div style="height: 100%; width: 100%; display: flex; justify-content: center; align-items: center">
    <span>
        Original Prompt Loaded.<br>
        Click "Apply" to apply the original prompt.
    </span>
</div>
"""
RECOMMEND_MARKDOWN = """
### Rcommended Model and Settings:

"""

tipo_path: pathlib.Path = model_path / "TIPO"

enable_tipo: bool = tipo_path.exists()

if not enable_tipo:
    logging.warning(
        "TIPO is not available. "
        f"Please download https://huggingface.co/KBlueLeaf/TIPO-500M-ft/blob/main/TIPO-500M-ft-F16.gguf into {tipo_path}"
    )


def apply_strength(
    tag_map: dict[str, str | list[str]],
    strength_map: dict[str, float],
    strength_map_nl: list[tuple[str, float]],
    break_map: set[str],
) -> dict[str, str | list[str]]:
    for cate in tag_map.keys():
        new_list = []
        # Skip natural language output at first
        if isinstance(tag_map[cate], str):
            # Ensure all the parts in the strength_map are in the prompt
            if all(part in tag_map[cate] for part, strength in strength_map_nl):
                org_prompt = tg.cast(str, tag_map[cate])
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, org_prompt = org_prompt.split(part, 1)
                    new_prompt += before.replace("(", r"\(").replace(")", r"\)")
                    part = part.replace("(", r"\(").replace(")", r"\)")
                    new_prompt += f"({part}:{strength})"
                new_prompt += org_prompt
            tag_map[cate] = new_prompt
            continue
        for org_tag in tag_map[cate]:
            tag = org_tag.replace("(", r"\(").replace(")", r"\)")
            if org_tag in strength_map:
                new_list.append(f"({tag}:{strength_map[org_tag]})")
            else:
                new_list.append(tag)
            if tag in break_map or org_tag in break_map:
                new_list.append("BREAK")
        tag_map[cate] = new_list

    return tag_map


def process(
    prompt: str,
    aspect_ratio: float,
    seed: int,
    tag_length: str,
    ban_tags: str,
    format: str,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    special_tags: list[str] = [],
    rating: list[str] = [],
    artist: list[str] = [],
    characters: list[str] = [],
    copyrights: list[str] = [],
    no_formatting: bool = False,
    nl_prompt: str = "",

) -> str:
    """process TIPO"""
    prompt = prompt.strip()
    seed = int(seed) % SEED_MAX

    propmt_preview = prompt.replace("\n", " ")[:40]
    logger.info(f"Processing propmt: {propmt_preview}...")
    logger.info(f"Processing with seed: {seed}")
    black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
    all_tags = [tag.strip().lower() for tag in prompt.strip().split(",") if tag.strip()]

    prompt_without_extranet, res = parse_prompt(prompt)
    prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)

    nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
    nl_prompt = ""
    strength_map_nl: list = []
    for part, strength in nl_prompt_parse_strength:
        nl_prompt += part
        if strength == 1:
            continue
        strength_map_nl.append((part, strength))

    rebuild_extranet = ""
    for name, params in res.items():
        for param in params:
            items = ":".join(param.items)
            rebuild_extranet += f" <{name}:{items}>"

    black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
    _tipo.BAN_TAGS = black_list
    all_tags = []
    strength_map = {}
    break_map = set()
    for part, strength in prompt_parse_strength:
        part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
        if part == "BREAK" and strength == -1:
            break_map.add(all_tags[-1])
            continue
        all_tags.extend(part_tags)
        if strength == 1:
            continue
        for tag in part_tags:
            strength_map[tag] = strength

    tag_length = tag_length.replace(" ", "_")
    nl_length = tag_length.replace(" ", "_")
    org_tag_map = seperate_tags(all_tags)

    # fill tag_map
    if special_tags:
        org_tag_map["special"].extend(special_tags)
    if rating:
        org_tag_map["rating"].extend(rating)
    if artist:
        org_tag_map["artist"].extend(artist)
    if characters:
        org_tag_map["characters"].extend(characters)
    if copyrights:
        org_tag_map["copyrights"].extend(copyrights)

    meta, operations, general, nl_prompt = parse_tipo_request(
        org_tag_map,
        nl_prompt,
        tag_length_target=tag_length,
        nl_length_target=nl_length,
        generate_extra_nl_prompt=(not nl_prompt and "<|extended|>" in format)
        or "<|generated|>" in format,
    )
    meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

    tag_map, _ = tipo_runner(
        meta,
        operations,
        general,
        nl_prompt,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        seed=seed,
    )

    addon: dict[str, str | list[str]] = {
        "tags": [],
        "nl": "",
    }
    for cate in tag_map.keys():
        if cate == "generated" and addon["nl"] == "":
            addon["nl"] = tag_map[cate]
            continue
        if cate == "extended":
            extended = tag_map[cate]
            addon["nl"] = extended
            continue
        if cate not in org_tag_map:
            continue
        for tag in tag_map[cate]:
            if tag in org_tag_map[cate]:
                continue
            tg.cast(list[str], addon["tags"]).append(tag)
    addon = apply_strength(addon, strength_map, strength_map_nl, break_map)
    unformatted_prompt_by_tipo = (
        prompt + ", " + ", ".join(addon["tags"]) + "\n" + tg.cast(str, addon["nl"])
    )
    tag_map = apply_strength(tag_map, strength_map, strength_map_nl, break_map)
    formatted_prompt_by_tipo = apply_format(tag_map, format).replace("BREAK,", "BREAK")

    if no_formatting:
        final_prompt = unformatted_prompt_by_tipo
    else:
        final_prompt = formatted_prompt_by_tipo

    result = final_prompt + "\n" + rebuild_extranet
    logger.info("Prompt processing done.")
    return result


@ft.lru_cache(maxsize=1024)
def tipo(
    text: str,
    seed: int,
    temperature: float = 0.5,
    top_p: float = 0.95,
    min_p: float = 0.05,
    top_k: int = 80,
    *,
    aspect_ratio: float = 1.0,
    target: str = TOTAL_TAG_LENGTH["LONG"],
    ban_tags: str = "",
    format: str = DEFAULT_FORMAT,
    special_tags: str = "",
    rating: str = "",
    artist: str = "",
    characters: str = "",
    copyrights: str = "",
    nl_prompt: str = "",
) -> str:
    """generate tags using tipo"""
    if not enable_tipo:
        return ""

    # or whatever path you want to put your model file
    models.model_dir = tipo_path

    # file = models.download_gguf(gguf_name="ggml-model-Q6_K.gguf")
    files = models.list_gguf()
    file = files[-1]
    logger.info(f"Use gguf model from local file: {file}")
    models.load_model(file, gguf=True, device="cpu")
    # models.load_model()
    # models.text_model.half().cuda()

    t0 = time.time_ns()
    result = process(
        text,
        aspect_ratio=aspect_ratio,
        seed=seed,
        tag_length=target,
        ban_tags=ban_tags,
        format=format,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        special_tags=[special_tags] if special_tags else [],
        rating=[rating] if rating else [],
        artist=[artist] if artist else [],
        characters=[characters] if characters else [],
        copyrights=[copyrights] if copyrights else [],
        nl_prompt=nl_prompt,
    )
    t1 = time.time_ns()
    logger.info(f"Result:\n{result}")
    logger.info(f"Time cost: {(t1 - t0) / 10**6:.1f}ms")

    return result
