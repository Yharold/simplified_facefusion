from front_end.core import register_ui_component
import gradio as gr

OUTPUT_IMAGE: gr.Image = None
OUTPUT_VIDEO: gr.Video = None


def render() -> None:
    global OUTPUT_IMAGE, OUTPUT_VIDEO
    OUTPUT_IMAGE = gr.Image(
        value=None, show_label=False, visible=True, interactive=False
    )
    OUTPUT_VIDEO = gr.Video(
        value=None, show_label=False, visible=False, interactive=False
    )
    register_ui_component("output_image", OUTPUT_IMAGE)
    register_ui_component("output_video", OUTPUT_VIDEO)


def listen() -> None:
    pass


def update() -> None:
    pass
