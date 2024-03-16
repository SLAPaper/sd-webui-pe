import functools as ft
import math
import typing as tg

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

import modules.options as options
import modules.shared as shared

from .utils import model_management, model_path, neg_inf, set_seed

expansion_path = model_path / "expansion"


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace("  ", " ")
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, "")
    return x


class PromptsExpansion:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            expansion_path, local_files_only=True
        )

        positive_words = (
            (expansion_path / "positive.txt").read_text(encoding="utf8").splitlines()
        )
        positive_words = ["Ġ" + x.lower() for x in positive_words if x != ""]

        self.logits_bias = (
            torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf
        )

        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])

        print(f"Prompt Expansion: Vocab with {len(debug_list)} words.")

        self.model = AutoModelForCausalLM.from_pretrained(
            expansion_path, local_files_only=True
        )
        self.model.eval()

    def logits_processor(self, input_ids, scores):
        with torch.inference_mode():
            assert scores.ndim == 2 and scores.shape[0] == 1
            self.logits_bias = self.logits_bias.to(scores)

            bias = self.logits_bias.clone()
            bias[0, input_ids[0].to(bias.device).long()] = neg_inf
            bias[0, 11] = 0

            return scores + bias

    @ft.lru_cache(maxsize=1024)
    def __call__(self, prompt: str, seed: int) -> str:
        if prompt == "":
            return ""

        prompt = safe_str(prompt) + ","
        set_seed(seed)

        with torch.inference_mode():
            tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
            tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(
                model_management.load_device
            )
            tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data[
                "attention_mask"
            ].to(model_management.load_device)

            current_token_length = int(tokenized_kwargs.data["input_ids"].shape[1])
            max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
            max_new_tokens = max_token_length - current_token_length

            opts = tg.cast(options.Options, shared.options)
            if (
                hasattr(opts, "Fooocus_V2_Max_New_Tokens")
                and opts.Fooocus_V2_Max_New_Tokens is not None
                and opts.Fooocus_V2_Max_New_Tokens > 0
            ):
                max_new_tokens = opts.Fooocus_V2_Max_New_Tokens

            model_management.load(self.model)
            # https://huggingface.co/blog/introducing-csearch
            # https://huggingface.co/docs/transformers/generation_strategies
            features = self.model.generate(
                **tokenized_kwargs,
                top_k=100,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                logits_processor=LogitsProcessorList([self.logits_processor]),
            )
            model_management.offload(self.model)

            response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
            result = safe_str(response[0])

            return result
