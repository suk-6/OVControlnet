import gradio as gr
from ovcontrolnet_tools import *

IRConversion()
ov_pipe = getOVPipe()
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"


def ovControlNet(image, prompt):
    pose = pose_estimator(image)
    result = ov_pipe(prompt, pose, 20, negative_prompt=negative_prompt)[0]
    return result, pose


ovDemo = gr.Interface(
    fn=ovControlNet,
    inputs=[gr.Image(width=512, height=512, type="numpy"), gr.Textbox(label="Prompt")],
    outputs=[
        gr.Image(label="Generated Image", type="numpy", show_label=True),
        gr.Image(label="Pose Estimation", type="numpy", show_label=True),
    ],
)

ovDemo.launch()
