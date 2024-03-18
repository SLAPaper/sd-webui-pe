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
                    with gr.Accordion("Advanced", open=False):
                        dtg_rating = gr.Radio(
                            [
                                "<|empty|>",
                                "safe",
                                "sensitive",
                                "nsfw",
                                "nsfw, explicit",
                            ],
                            value="<|empty|>",
                            label="DanTagGen beta Rating",
                        )
                        dtg_artist = gr.Textbox(
                            value="", label="DanTagGen beta Artist", placeholder=""
                        )
                        dtg_chara = gr.Textbox(
                            value="", label="DanTagGen beta Characters", placeholder=""
                        )
                        dtg_copy = gr.Textbox(
                            value="",
                            label="DanTagGen beta Copyrights(Series)",
                            placeholder="",
                        )
                        dtg_target = gr.Radio(
                            ["very_short", "short", "long", "very_long"],
                            value="long",
                            label="DanTagGen beta Target length",
                        )

        return [
            is_enabled,
            model_selection,
            discard_original,
            dtg_rating,
            dtg_artist,
            dtg_chara,
            dtg_copy,
            dtg_target,
        ]

    def process(self, p: StableDiffusionProcessing, *args) -> None:
        is_enabled: bool = args[0]
        if not is_enabled:
            return

        model_selection: str = args[1]
        discard_original: bool = args[2]
        dtg_rating: str = args[3]
        dtg_artist: str = args[4]
        dtg_chara: str = args[5]
        dtg_copy: str = args[6]
        dtg_target: str = args[7]

        for i, prompt in enumerate(p.all_prompts):
            if model_selection == "Fooocus V2":
                positivePrompt = expansion(prompt, p.all_seeds[i])
            elif model_selection == "SuperPrompt v1":
                sp = super_prompt(prompt, p.all_seeds[i])
                if discard_original:
                    positivePrompt = sp
                else:
                    positivePrompt = f"{prompt}, BREAK, {sp}"
            elif model_selection == "DanTagGen-beta":
                dtg = dtg_beta(
                    prompt,
                    p.all_seeds[i],
                    rating=dtg_rating if dtg_rating else "<|empty|>",
                    artist=dtg_artist if dtg_artist else "<|empty|>",
                    characters=dtg_chara if dtg_chara else "<|empty|>",
                    copyrights=dtg_copy if dtg_copy else "<|empty|>",
                    aspect_ratio=p.width / p.height,
                    target=dtg_target if dtg_target else "long",
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
            label="Max new token length for Fooocus V2",
            infotext="Set to 0 to fill up remaining tokens of 75*k",
            component=gr.Slider,
            component_args={
                "minimum": 0,
                "maximum": 300,
                "step": 1,
            },
            section=section,
            onchange=lambda: expansion.__call__.cache_clear(),
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
                "maximum": 300,
                "step": 75,
            },
            section=section,
            onchange=lambda: super_prompt.cache_clear(),
        ),
    )
    opts.add_option(
        "DanTagGen_beta_Max_New_Tokens",
        shared.OptionInfo(
            default=0,
            label="Max token length for DanTagGen-beta",
            infotext="Set to 0 to fill up remaining tokens of 75*k",
            component=gr.Slider,
            component_args={
                "minimum": 0,
                "maximum": 300,
                "step": 1,
            },
            section=section,
            onchange=lambda: dtg_beta.cache_clear(),
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
