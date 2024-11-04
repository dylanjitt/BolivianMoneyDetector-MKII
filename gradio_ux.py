import gradio as gr
from src.detector import LLM, BilleteDetector

# Initialize your detector and language model instances
detector = BilleteDetector()
llm = LLM()

def process_image(img):
    # Process image using BilleteDetector's `showImg` method
    new_saldo, processed_img = detector.showImg(img)
    return new_saldo, processed_img

def generate_text_response(spanish, voice):
    # Generate a text and optionally an audio response using LLM
    response_text, audio_path = llm.generate_response(spanish, voice)
    return response_text, audio_path  # Separate response text and audio path

def toggle_audio_visibility(enable_voice):
    return gr.update(visible=enable_voice)

# Gradio UI setup
def main():
    with gr.Blocks() as app:
        # Main menu display
        with gr.Column(visible=True) as menu:
            saldo = gr.Number(value=0.00, label="Saldo disponible", interactive=False, precision=2)

            with gr.Row():
                image_input = gr.Image(type="filepath", label="Drag and drop an image")
                image_output = gr.Image(label="Processed Image")  # To display processed image
                gr.Button("Process Image").click(
                    fn=process_image,
                    inputs=[image_input],
                    outputs=[saldo, image_output]
                )

            response = gr.Textbox(label="Response", interactive=False)
            audio_output = gr.Audio(label="Generated Audio", visible=False)
            
            with gr.Row():
                voice = gr.Checkbox(label="Enable Voice", value=False)
                spanish = gr.Checkbox(label="Translate to Spanish", value=False)
                
                gr.Button("Checkout").click(
                    fn=generate_text_response,
                    inputs=[spanish, voice],
                    outputs=[response, audio_output]
                )

            # Toggle visibility of audio output based on checkbox
            voice.change(fn=toggle_audio_visibility, inputs=voice, outputs=audio_output)

    return app

# Launch the app
app = main()
app.launch(share=True)#debug=True)

