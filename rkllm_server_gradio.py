import sys
import resource
import subprocess
import gradio as gr
from ctypes_bindings import *
from model_class import *
from mesh_utils import *

# Set environment variables
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "8080"
os.environ["RKLLM_LOG_LEVEL"] = "1"
# Set resource limit
resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

history = []

if __name__ == "__main__":


    # Setting a fixed frequency
    command = "sudo bash ./lib/fix_freq_rk3588.sh"
    subprocess.run(command, shell=True)

    # Setting a limit on the number of file descriptors
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Helper function to define initializing model before class is declared
    # Without this, you would need to initialize the class before you select the model
    def initialize_model(model):
        global rkllm_model
        # Have to unload previous model in single-threaded mode
        try:
            rkllm_model.release()
        except:
            print("No model loaded! Continuing with initialization...")
        # Initialize RKLLM model
        init_msg = "=========INITIALIZING==========="
        print(init_msg)
        sys.stdout.flush()
        rkllm_model = RKLLMLoaderClass(model=model)
        model_init = f"RKLLM Model, {rkllm_model.model_name} has been initialized successfully！"
        print(model_init)
        complete_init = "=============================="
        print(complete_init)
        output = [[f"<h4 style=\"text-align:center;\">{model_init}\n</h4>", None]]
        sys.stdout.flush()
        return output 
    # Helper function to stream LLM output into the chat box
    def get_RKLLM_output(message, history):
        try:
            yield from rkllm_model.get_RKLLM_output(message, history)
        except RuntimeError as e:
            print(f"ERROR: {e}")
        return history

    # Create a Gradio interface
    with gr.Blocks(title="Chat with RKLLM") as chatRKLLM:
        available_models = available_models()
        gr.Markdown("<div align='center'><font size='10'> Definitely Not Skynet </font></div>")
        with gr.Tabs():
            with gr.TabItem("Select Model"):
                model_dropdown = gr.Dropdown(choices=available_models, label="Select Model", value="None", allow_custom_value=True)
                statusBox = gr.Chatbot(height=100)
                model_dropdown.input(initialize_model, [model_dropdown], [statusBox])
            with gr.TabItem("Txt2Txt"):
                txt2txt = gr.ChatInterface(fn=get_RKLLM_output, type="messages")
                txt2txt.chatbot.height = "70vh"
                txt2txt.chatbot.resizeable = True
            with gr.TabItem("Txt2Mesh"):
                with gr.Row():    
                    with gr.Column(scale=2):
                        txt2txt = gr.ChatInterface(fn=get_RKLLM_output, type="messages")
                        txt2txt.chatbot.height = "70vh"
                        txt2txt.chatbot.resizeable = True
                    with gr.Column(scale=2):
                        # Add the text box for 3D mesh input and button
                        mesh_input = gr.Textbox(
                            label="3D Mesh Input",
                            placeholder="Paste your 3D mesh in OBJ format here...",
                            lines=5,
                        )
                        visualize_button = gr.Button("Visualize 3D Mesh")
                        output_model = gr.Model3D(
                                    label="3D Mesh Visualization",
                                    interactive=False,
                                )
                        # Link the button to the visualization function
                        visualize_button.click(
                            fn=apply_gradient_color,
                            inputs=[mesh_input],
                            outputs=[output_model]
                            )
        print("\nNo model loaded yet!\n")


    # Enable the event queue system.
    chatRKLLM.queue()
    # Start the Gradio application.
    chatRKLLM.launch()

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")
