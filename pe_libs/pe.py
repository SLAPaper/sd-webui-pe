import functools as ft
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

# Import the required modules
from ldm_patched.modules import model_management
from ldm_patched.modules.model_patcher import ModelPatcher

from .utils import model_path, neg_inf, set_seed

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

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()

        # MPS hack
        if model_management.is_device_mps(load_device):
            load_device = torch.device("cpu")
            offload_device = torch.device("cpu")

        use_fp16 = model_management.should_use_fp16(device=load_device)

        if use_fp16:
            self.model.half()

        self.patcher = ModelPatcher(
            self.model, load_device=load_device, offload_device=offload_device
        )
        print(
            f"Prompt Expansion engine loaded for {load_device}, use_fp16 = {use_fp16}."
        )

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

        if self.patcher.current_device != self.patcher.load_device:
            print("Prompt Expansion loaded by itself.")
            model_management.load_model_gpu(self.patcher)

        prompt = safe_str(prompt) + ","
        set_seed(seed)

        with torch.inference_mode():
            tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
            tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(
                self.patcher.load_device
            )
            tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data[
                "attention_mask"
            ].to(self.patcher.load_device)

            current_token_length = int(tokenized_kwargs.data["input_ids"].shape[1])
            max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
            max_new_tokens = max_token_length - current_token_length

            # https://huggingface.co/blog/introducing-csearch
            # https://huggingface.co/docs/transformers/generation_strategies
            features = self.model.generate(
                **tokenized_kwargs,
                top_k=100,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                logits_processor=LogitsProcessorList([self.logits_processor]),
            )

            response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
            result = safe_str(response[0])

            return result
