from typing import Any
from front_end.core import register_ui_component
import gradio as gr
from mid_end import globals
from front_end import choices_value, wording

PROCESS_TOOL: gr.Dropdown = None
PROCESS_TOOL_DESCRIBE: gr.HTML = None


def render() -> None:
    global PROCESS_TOOL, PROCESS_TOOL_DESCRIBE
    gr.Markdown("# 工具选择<br>")
    PROCESS_TOOL = gr.Dropdown(
        choices=choices_value.tool_choices,
        value=choices_value.tool_choices[0],
        interactive=True,
    )
    gr.Markdown("## 工具描述")
    PROCESS_TOOL_DESCRIBE = gr.HTML(value="<p> something </p><br>other something")
    register_ui_component("process_tool", PROCESS_TOOL)


def listen() -> None:
    PROCESS_TOOL.change(update, inputs=PROCESS_TOOL, outputs=PROCESS_TOOL_DESCRIBE)


def update(choice: Any) -> gr.HTML:
    if choice == choices_value.tool_choices[0]:
        globals.frame_processor = "face_debugger"
        return gr.HTML(value=wording.get("debugger_describe"))
    if choice == choices_value.tool_choices[1]:
        globals.frame_processor = "face_swapper"
        return gr.HTML(value=wording.get("swapper_describe"))
    if choice == choices_value.tool_choices[2]:
        globals.frame_processor = "frame_enhancer"
        return gr.HTML(value=wording.get("enhancer_describe"))
    if choice == choices_value.tool_choices[3]:
        globals.frame_processor = "frame_colorizer"
        return gr.HTML(value=wording.get("colorizer_describe"))
