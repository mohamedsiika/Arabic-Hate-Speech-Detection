import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your best model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./FT_MARBERT").to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("./FT_MARBERT")

# Create inference pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

label_arabic_map = {
    "LABEL_0": "غير مسيء",
    "LABEL_1": "مسيء"
}

# Define prediction function
def predict(text):
    result = pipe(text, truncation=True, max_length=256)
    label = result[0]['label']
    score = result[0]['score']
    return f"{label_arabic_map[label]}  |  الثقة :  {score:.2f}"

# Launch GUI
gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Hate Speech Detector"
).launch(share=False, inbrowser=True)
