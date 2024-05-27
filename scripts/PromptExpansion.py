import typing as tg

import gradio as gr
import gradio.components.base as gr_base
from pe_libs.dtg import dtg as _dtg
from pe_libs.pe import PromptsExpansion
from pe_libs.super_prompt import super_prompt

from modules import options, script_callbacks, scripts, shared
from modules.processing import StableDiffusionProcessing
from modules.ui_components import FormColumn, FormRow

expansion = PromptsExpansion()


class PromptExpansion(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self) -> str:
        return "Prompt Expansion 1.0"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> list:
        with gr.Group():
            with gr.Accordion("Prompt Expansion", open=False):
                with FormRow():
                    with FormColumn(min_width=160):
                        is_enabled = gr.Checkbox(
                            value=False,
                            label="Enable Prompt Expansion",
                            info="Use extra model to expand prompts",
                        )
                with FormRow():
                    model_selection = gr.Radio(
                        choices=[
                            "Fooocus V2",
                            "SuperPrompt v1",
                            "DanTagGen",
                        ],
                        value="Fooocus V2",
                        label="Model use for prompt expansion",
                    )
                with FormRow():
                    with FormColumn(min_width=160):
                        discard_original = gr.Checkbox(
                            value=False,
                            label="Discard original prompts in SuperPrompt V1",
                            info="otherwise will append to original prompts",
                        )
                with FormRow():
                    with gr.Accordion("Advanced Options", open=False):
                        fv2_topk = gr.Number(
                            value=100,
                            label="Fooocus V2 TopK",
                            min_value=1,
                            placeholder=(
                                "TopK for Fooocus V2, the smaller the less randomness"
                            ),
                        )
                        sp_prompt = gr.Textbox(
                            value="",
                            label="SuperPrompt v1 Custom Prompt",
                            placeholder=(
                                "Custom Prompt for SuperPrompt v1, "
                                "original prompt is appended followed, "
                                "leave empty to use default"
                            ),
                        )
                        sp_temperature = gr.Number(
                            value=0.7,
                            label="SuperPrompt v1 Temperature",
                            min_value=0.0,
                            max_value=2.0,
                            placeholder=(
                                "Temperature for SuperPrompt v1, "
                                "the higher the more randomness"
                            ),
                        )
                        sp_topk = gr.Number(
                            value=50,
                            label="SuperPrompt v1 TopK",
                            min_value=1,
                            placeholder=(
                                "TopK for SuperPrompt v1, the smaller the less randomness"
                            ),
                        )
                        sp_topp = gr.Number(
                            value=0.95,
                            label="SuperPrompt v1 TopP",
                            min_value=0.0,
                            max_value=1.0,
                            placeholder=(
                                "TopP for SuperPrompt v1, "
                                "the higher the more randomness"
                            ),
                        )
                        dtg_temperature = gr.Number(
                            value=1.35,
                            label="DanTagGen Temperature",
                            min_value=0.0,
                            max_value=2.0,
                            placeholder=(
                                "Temperature for DanTagGen, "
                                "the higher the more randomness"
                            ),
                        )
                        dtg_topk = gr.Number(
                            value=100,
                            label="DanTagGen TopK",
                            min_value=1,
                            placeholder=(
                                "TopK for DanTagGen, the smaller the less randomness"
                            ),
                        )
                        dtg_topp = gr.Number(
                            value=0.95,
                            label="DanTagGen TopP",
                            min_value=0.0,
                            max_value=1.0,
                            placeholder=(
                                "TopP for DanTagGen, " "the higher the more randomness"
                            ),
                        )
                        dtg_repeat_penalty = gr.Number(
                            value=1.17,
                            label="DanTagGen Repeat Penalty",
                            min_value=0.0,
                            max_value=2.0,
                            placeholder=(
                                "Repeat penalty for DanTagGen, "
                                "the higher the less repeated tokens"
                            ),
                        )
                        dtg_rating = gr.Radio(
                            [
                                "<|empty|>",
                                "safe",
                                "sensitive",
                                "nsfw",
                                "nsfw, explicit",
                            ],
                            value="<|empty|>",
                            label=(
                                "DanTagGen Rating, "
                                "use <|empty|> for no rating tendency"
                            ),
                        )
                        dtg_artist = gr.Textbox(
                            value="",
                            label="DanTagGen Artist",
                            placeholder=(
                                "Artist tag for DanTagGen, "
                                "leave empty to use <|empty|>"
                            ),
                        )
                        dtg_chara = gr.Textbox(
                            value="",
                            label="DanTagGen Characters",
                            placeholder=(
                                "Characters tag for DanTagGen, "
                                "leave empty to use <|empty|>"
                            ),
                        )
                        dtg_copy = gr.Textbox(
                            value="",
                            label="DanTagGen Copyrights(Series)",
                            placeholder=(
                                "Copyrights(Series) tag for DanTagGen, "
                                "leave empty to use <|empty|>"
                            ),
                        )
                        dtg_target = gr.Radio(
                            ["very_short", "short", "long", "very_long"],
                            value="long",
                            label=(
                                "DanTagGen Target length, "
                                "short or long is recommended"
                            ),
                        )
                        dtg_banned = gr.Textbox(
                            value="",
                            label="DanTagGen banned tags",
                            placeholder=(
                                "Banned tags for DanTagGen, seperated by comma, case insensitive"
                            ),
                        )

        return [
            is_enabled,
            model_selection,
            discard_original,
            fv2_topk,
            sp_prompt,
            sp_temperature,
            sp_topk,
            sp_topp,
            dtg_temperature,
            dtg_topk,
            dtg_topp,
            dtg_repeat_penalty,
            dtg_rating,
            dtg_artist,
            dtg_chara,
            dtg_copy,
            dtg_target,
            dtg_banned,
        ]

    def process(self, p: StableDiffusionProcessing, *args) -> None:
        is_enabled: bool = args[0]
        if not is_enabled:
            return

        model_selection: str = args[1]
        discard_original: bool = args[2]

        fv2_topk: int = int(args[3])

        sp_prompt: str = args[4]
        sp_temperature: float = args[5]
        sp_topk: int = int(args[6])
        sp_topp: float = args[7]

        dtg_temperature: float = args[8]
        dtg_topk: int = int(args[9])
        dtg_topp: float = args[10]
        dtg_repeat_penalty: float = args[11]
        dtg_rating: str = args[12]
        dtg_artist: str = args[13]
        dtg_chara: str = args[14]
        dtg_copy: str = args[15]
        dtg_target: str = args[16]
        dtg_banned: str = args[17]

        opts = tg.cast(options.Options, shared.opts)
        max_new_tokens: int = 0
        match model_selection:
            case "Fooocus V2":
                if (
                    "Fooocus_V2_Max_New_Tokens" in opts.data
                    and opts.data["Fooocus_V2_Max_New_Tokens"] is not None
                    and int(str(opts.data["Fooocus_V2_Max_New_Tokens"])) > 0
                ):
                    max_new_tokens = int(str(opts.data["Fooocus_V2_Max_New_Tokens"]))
            case "SuperPrompt v1":
                if (
                    "SuperPrompt_V1_Max_Tokens" in opts.data
                    and opts.data["SuperPrompt_V1_Max_Tokens"] is not None
                    and int(str(opts.data["SuperPrompt_V1_Max_Tokens"])) > 0
                ):
                    max_new_tokens = int(str(opts.data["SuperPrompt_V1_Max_Tokens"]))
            case "DanTagGen":
                if (
                    "DanTagGen_Max_New_Tokens" in opts.data
                    and opts.data["DanTagGen_Max_New_Tokens"] is not None
                    and int(str(opts.data["DanTagGen_Max_New_Tokens"])) > 0
                ):
                    max_new_tokens = int(str(opts.data["DanTagGen_Max_New_Tokens"]))
            case _:
                raise NotImplementedError(f"Model {model_selection} not implemented")

        # print(f"DEBUG: {model_selection=}, {max_new_tokens=}")

        for i, prompt in enumerate(p.all_prompts):
            match model_selection:
                case "Fooocus V2":
                    positivePrompt = expansion(
                        prompt, p.all_seeds[i], max_new_tokens, top_k=fv2_topk
                    )
                case "SuperPrompt v1":
                    sp = super_prompt(
                        prompt,
                        p.all_seeds[i],
                        max_new_tokens,
                        sp_prompt,
                        temperature=sp_temperature,
                        top_k=sp_topk,
                        top_p=sp_topp,
                    )
                    if discard_original:
                        positivePrompt = sp
                    else:
                        positivePrompt = f"{prompt}, BREAK, {sp}"
                case "DanTagGen":
                    dtg = _dtg(
                        prompt,
                        p.all_seeds[i],
                        max_new_tokens,
                        temperature=dtg_temperature,
                        top_k=dtg_topk,
                        top_p=dtg_topp,
                        repetition_penalty=dtg_repeat_penalty,
                        rating=dtg_rating if dtg_rating else "<|empty|>",
                        artist=dtg_artist if dtg_artist else "<|empty|>",
                        characters=dtg_chara if dtg_chara else "<|empty|>",
                        copyrights=dtg_copy if dtg_copy else "<|empty|>",
                        aspect_ratio=p.width / p.height,
                        target=dtg_target if dtg_target else "long",
                        banned_tags=dtg_banned,
                    )
                    positivePrompt = f"{prompt}, {dtg}"
                case _:
                    raise NotImplementedError(
                        f"Model {model_selection} not implemented"
                    )

            p.all_prompts[i] = positivePrompt

        p.extra_generation_params["Prompt-Expansion"] = True

        match model_selection:
            case _:
                p.extra_generation_params["Prompt-Expansion-Model"] = model_selection

    def after_component(self, component: gr_base.Component, **kwargs) -> None:
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/7456#issuecomment-1414465888 helpfull link
        # Find the text2img textbox component
        if kwargs.get("elem_id") == "txt2img_prompt":  # postive prompt textbox
            self.boxx = component
        # Find the img2img textbox component
        if kwargs.get("elem_id") == "img2img_prompt":  # postive prompt textbox
            self.boxxIMG = component


def on_ui_settings():
    section = ("Prompt_Expansion", "Prompt Expansion")

    opts = tg.cast(options.Options, shared.opts)
    opts.add_option(
        "Fooocus_V2_Max_New_Tokens",
        shared.OptionInfo(
            default=0,
            label="Max new token length for Fooocus V2 (Set to 0 to fill up remaining tokens of 75*k)",
            component=gr.Slider,
            component_args={
                "minimum": 0,
                "maximum": 300,
                "step": 1,
            },
            section=section,
        ),
    )
    opts.add_option(
        "SuperPrompt_V1_Max_Tokens",
        shared.OptionInfo(
            default=150,
            label="Max token length for SuperPrompt V1",
            component=gr.Slider,
            component_args={
                "minimum": 75,
                "maximum": 375,
                "step": 75,
            },
            section=section,
        ),
    )
    opts.add_option(
        "DanTagGen_Max_New_Tokens",
        shared.OptionInfo(
            default=0,
            label="Max new token length for DanTagGen (Set to 0 to fill up remaining tokens of 75*k)",
            component=gr.Slider,
            component_args={
                "minimum": 0,
                "maximum": 300,
                "step": 1,
            },
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
