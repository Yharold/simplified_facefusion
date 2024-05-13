from front_end.core import get_ui_component
import gradio as gr
from front_end import wording, choices_value
from mid_end import globals

PREVIEW_IMAGE: gr.Image = None


def render() -> None:
    global PREVIEW_IMAGE
    gr.Markdown("## 效果展示<br>")
    PREVIEW_IMAGE = gr.Image(
        show_label=False,
        interactive=False,
        value=choices_value.preview_image_path["face_debugger"],
    )


def listen() -> None:
    process_tool = get_ui_component("process_tool")
    process_tool.change(update_preview, inputs=process_tool, outputs=PREVIEW_IMAGE)


def update_preview(choice: str) -> gr.Image:
    idx = choices_value.tool_choices.index(choice)
    if idx == 0:
        image_path = choices_value.preview_image_path["face_debugger"]
    if idx == 1:
        image_path = choices_value.preview_image_path["face_swapper"]
    if idx == 2:
        image_path = choices_value.preview_image_path["frame_enhancer"]
    if idx == 3:
        image_path = choices_value.preview_image_path["frame_colorizer"]
    return gr.Image(
        show_label=False,
        interactive=False,
        value=image_path,
    )
