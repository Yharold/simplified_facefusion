import multiprocessing
from typing import Any
import gradio as gr
from front_end.components import (
    about,
    process,
    preview,
    source,
    target,
    reference,
    output,
    process_button,
)


def pre_render() -> bool:
    return True


def render() -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Blocks():
                    about.render()
                with gr.Blocks():
                    process.render()
                    process.listen()
            with gr.Column(scale=4):
                with gr.Blocks():
                    preview.render()
                    preview.listen()
                with gr.Blocks():
                    target.render()
                    target.listen()
                with gr.Blocks():
                    reference.render()
                    reference.listen()
                with gr.Blocks():
                    source.render()
                    source.listen()
                with gr.Blocks():
                    output.render()
                    output.listen()
                with gr.Blocks():
                    process_button.render()
                    process_button.listen()


def listen() -> None:
    pass


def run(ui: gr.Blocks) -> None:
    concurrency_count = min(8, multiprocessing.cpu_count())
    ui.queue(concurrency_count=concurrency_count).launch(
        show_api=False, quiet=True, server_port=4396
    )
