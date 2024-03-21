import typing as tg

import gradio as gr
import gradio.components.base as gr_base
from pe_libs.pe import PromptsExpansion
from pe_libs.super_prompt import super_prompt
from pe_libs.dtg_beta import dtg_beta

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
                        choices=["Fooocus V2", "SuperPrompt v1", "DanTagGen-beta"],
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
                        sp_prompt = gr.Textbox(
                            value="",
                            label="SuperPrompt v1 Custom Prompt",
                            placeholder=(
                                "Custom Prompt for SuperPrompt v1, "
                                "original prompt is appended followed, "
                                "leave empty to use default"
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
                                "DanTagGen beta Rating, "
                                "use <|empty|> for no rating tendency"
                            ),
                        )
                        dtg_artist = gr.Textbox(
                            value="",
                            label="DanTagGen beta Artist",
                            placeholder=(
                                "Artist tag for DanTagGen beta, "
                                "leave empty to use <|empty|>"
                            ),
                        )
                        dtg_chara = gr.Textbox(
                            value="",
                            label="DanTagGen beta Characters",
                            placeholder=(
                                "Characters tag for DanTagGen beta, "
                                "leave empty to use <|empty|>"
                            ),
                        )
                        dtg_copy = gr.Textbox(
                            value="",
                            label="DanTagGen beta Copyrights(Series)",
                            placeholder=(
                                "Copyrights(Series) tag for DanTagGen beta, "
                                "leave empty to use <|empty|>"
                            ),
                        )
                        dtg_target = gr.Radio(
                            ["very_short", "short", "long", "very_long"],
                            value="long",
                            label=(
                                "DanTagGen beta Target length, "
                                "short or long is recommended"
                            ),
                        )
                        dtg_banned = gr.Textbox(
                            value="",
                            label="DanTagGen beta banned tags",
                            placeholder=(
                                "Banned tags for DanTagGen beta, seperated by comma"
                            ),
                        )

        return [
            is_enabled,
            model_selection,
            discard_original,
            sp_prompt,
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
        sp_prompt: str = args[3]
        dtg_rating: str = args[4]
        dtg_artist: str = args[5]
        dtg_chara: str = args[6]
        dtg_copy: str = args[7]
        dtg_target: str = args[8]
        dtg_banned: str = args[9]

        opts = tg.cast(options.Options, shared.opts)
        max_new_tokens = 0
        if model_selection == "Fooocus V2":
            if (
                "Fooocus_V2_Max_New_Tokens" in opts.data
                and opts.data["Fooocus_V2_Max_New_Tokens"] is not None
                and opts.data["Fooocus_V2_Max_New_Tokens"] > 0
            ):
                max_new_tokens = opts.data["Fooocus_V2_Max_New_Tokens"]
        elif model_selection == "SuperPrompt v1":
            if (
                "SuperPrompt_V1_Max_Tokens" in opts.data
                and opts.data["SuperPrompt_V1_Max_Tokens"] is not None
                and opts.data["SuperPrompt_V1_Max_Tokens"] > 0
            ):
                max_new_tokens = opts.data["SuperPrompt_V1_Max_Tokens"]
        elif model_selection == "DanTagGen-beta":
            if (
                "DanTagGen_beta_Max_New_Tokens" in opts.data
                and opts.data["DanTagGen_beta_Max_New_Tokens"] is not None
                and opts.data["DanTagGen_beta_Max_New_Tokens"] > 0
            ):
                max_new_tokens = opts.data["DanTagGen_beta_Max_New_Tokens"]
        else:
            raise NotImplementedError(f"Model {model_selection} not implemented")

        # print(f"DEBUG: {model_selection=}, {max_new_tokens=}")

        for i, prompt in enumerate(p.all_prompts):
            if model_selection == "Fooocus V2":
                positivePrompt = expansion(prompt, p.all_seeds[i], max_new_tokens)
            elif model_selection == "SuperPrompt v1":
                sp = super_prompt(prompt, p.all_seeds[i], max_new_tokens, sp_prompt)
                if discard_original:
                    positivePrompt = sp
                else:
                    positivePrompt = f"{prompt}, BREAK, {sp}"
            elif model_selection == "DanTagGen-beta":
                dtg = dtg_beta(
                    prompt,
                    p.all_seeds[i],
                    max_new_tokens,
                    rating=dtg_rating if dtg_rating else "<|empty|>",
                    artist=dtg_artist if dtg_artist else "<|empty|>",
                    characters=dtg_chara if dtg_chara else "<|empty|>",
                    copyrights=dtg_copy if dtg_copy else "<|empty|>",
                    aspect_ratio=p.width / p.height,
                    target=dtg_target if dtg_target else "long",
                    banned_tags=dtg_banned,
                )
                positivePrompt = f"{prompt}, {dtg}"
            else:
                raise NotImplementedError(f"Model {model_selection} not implemented")

            p.all_prompts[i] = positivePrompt

        p.extra_generation_params["Prompt-Expansion"] = True
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
        "DanTagGen_beta_Max_New_Tokens",
        shared.OptionInfo(
            default=0,
            label="Max new token length for DanTagGen-beta (Set to 0 to fill up remaining tokens of 75*k)",
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
