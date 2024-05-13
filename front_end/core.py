import os, sys, importlib
from typing import Any, Dict, List, Optional
import gradio as gr
from mid_end.vision import get_temp_path
from mid_end import logger

os.environ["GRADIO_TEMP_DIR"] = get_temp_path()
UI_COMPONENTS: Dict[str, Any] = {}


def register_ui_component(component_name: str, component: Any) -> None:
    UI_COMPONENTS[component_name] = component


def get_ui_component(component_name: str) -> Any:
    if component_name in UI_COMPONENTS:
        return UI_COMPONENTS[component_name]
    return None


def load_layout_module() -> Any:
    try:
        layout_module = importlib.import_module("front_end.default")
    except ModuleNotFoundError as exception:
        logger.error("default layout not loaded!", __name__.upper())
        logger.debug(exception.msg, __name__.upper())
        sys.exit(1)
    return layout_module


def launch() -> None:
    with gr.Blocks(theme=get_theme(), title="simplified facefusion") as ui:
        layout_module = load_layout_module()
        if layout_module.pre_render():
            layout_module.render()
            layout_module.listen()
    if layout_module:
        layout_module.run(ui)


def get_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.neutral,
        font=gr.themes.GoogleFont("Open Sans"),
    ).set(
        background_fill_primary="*neutral_100",
        block_background_fill="white",
        block_border_width="0",
        block_label_background_fill="*primary_100",
        block_label_background_fill_dark="*primary_600",
        block_label_border_width="none",
        block_label_margin="0.5rem",
        block_label_radius="*radius_md",
        block_label_text_color="*primary_500",
        block_label_text_color_dark="white",
        block_label_text_weight="600",
        block_title_background_fill="*primary_100",
        block_title_background_fill_dark="*primary_600",
        block_title_padding="*block_label_padding",
        block_title_radius="*block_label_radius",
        block_title_text_color="*primary_500",
        block_title_text_size="*text_sm",
        block_title_text_weight="600",
        block_padding="0.5rem",
        border_color_primary="transparent",
        border_color_primary_dark="transparent",
        button_large_padding="2rem 0.5rem",
        button_large_text_weight="normal",
        button_primary_background_fill="*primary_500",
        button_primary_text_color="white",
        button_secondary_background_fill="white",
        button_secondary_border_color="transparent",
        button_secondary_border_color_dark="transparent",
        button_secondary_border_color_hover="transparent",
        button_secondary_border_color_hover_dark="transparent",
        button_secondary_text_color="*neutral_800",
        button_small_padding="0.75rem",
        checkbox_background_color="*neutral_200",
        checkbox_background_color_selected="*primary_600",
        checkbox_background_color_selected_dark="*primary_700",
        checkbox_border_color_focus="*primary_500",
        checkbox_border_color_focus_dark="*primary_600",
        checkbox_border_color_selected="*primary_600",
        checkbox_border_color_selected_dark="*primary_700",
        checkbox_label_background_fill="*neutral_50",
        checkbox_label_background_fill_hover="*neutral_50",
        checkbox_label_background_fill_selected="*primary_500",
        checkbox_label_background_fill_selected_dark="*primary_600",
        checkbox_label_text_color_selected="white",
        input_background_fill="*neutral_50",
        shadow_drop="none",
        slider_color="*primary_500",
        slider_color_dark="*primary_600",
    )
