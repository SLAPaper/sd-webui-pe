import typing as tg

import gradio as gr
import gradio.components.base as gr_base
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
        return "Prompt-Expansion 1.0"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> list:
        with gr.Group():
            with gr.Accordion("Prompt-Expansion", open=False):
                with FormRow():
                    with FormColumn(min_width=160):
                        is_enabled = gr.Checkbox(
                            value=False,
                            label="Enable Prompt Expansion",
                            info="Use extra model to expand prompts",
                        )
                with FormRow():
                    model_selection = gr.Radio(
                        choices=["Fooocus V2", "SuperPrompt v1"],
                        value="Fooocus V2",
                        label="Model use for prompt expansion",
                    )

        return [is_enabled, model_selection]

    def process(self, p: StableDiffusionProcessing, *args) -> None:
        is_enabled: bool = args[0]
        if not is_enabled:
            return

        model_selection: str = args[1]

        for i, prompt in enumerate(p.all_prompts):
            positivePrompt = (
                expansion(prompt, p.all_seeds[i])
                if model_selection == "Fooocus V2"
                else f"{prompt}, BREAK, {super_prompt(prompt, p.all_seeds[i])}"
            )

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


script_callbacks.on_ui_settings(on_ui_settings)
