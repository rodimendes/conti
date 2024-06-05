import time
import gradio as gr
import torch

from transformers import pipeline

def ask_model(prompt, model, max_tokens=100):

    start_time = time.time()

    generate_text = pipeline(
        model=model, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        device_map="auto"
    )
    
    result = generate_text(
        prompt,
        max_new_tokens=max_tokens
    )

    answer = result[0]["generated_text"]

    elapsed_time = time.time() - start_time

    return answer, elapsed_time

if __name__ == "__main__":

    # available_models = ["databricks/dolly-v2-7b"]
    available_models = ["databricks/dolly-v2-7b"]
    
    demo = gr.Interface(fn=ask_model, 
                        inputs=["text", gr.Dropdown(available_models, label="model", value="databricks/dolly-v2-7b")], 
                        outputs=["text", "number"],
                        title="ContiGPT")
    demo.launch(share=True)