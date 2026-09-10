import gradio as gr
from fastapi import FastAPI

from gradio_dashboard import dashboard

app = FastAPI()

app = gr.mount_gradio_app(
    app,
    dashboard,
    path="/"
)
