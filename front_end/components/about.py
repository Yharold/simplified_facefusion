from mid_end import metadata
import gradio as gr

ABOUT_BUTTON: gr.Button = None


def render() -> None:
    global ABOUT_BUTTON
    ABOUT_BUTTON = gr.Button(metadata.get("name"), variant="primary", size="sm")
