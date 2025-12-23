import torch
import torch.nn as nn
import gradio as gr
import pickle
import re
import unicodedata

# LABELS 
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# CLEAN TEXT 
def clean_text(text):
    text = text.lower()

    text = unicodedata.normalize("NFKC", text)
    
    # remove URLs
    text = re.sub(r"http\S+", "", text)
    
    text = re.sub(r"[^a-z!?*\s]", " ", text)
    
    text = re.sub(r"([!?*]){2,}", r"\1", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

# MODEL DEFINITION 
class ToxicMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 6)
        )

    def forward(self, x):
        return self.net(x)

#LOAD VECTORIZER 
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# LOAD MODEL 
model = ToxicMLP(input_dim=len(vectorizer.get_feature_names_out()))
model.load_state_dict(
    torch.load("best_model_toxic_cmt_clss.pt", map_location="cpu")
)
model.eval()

# PREDICT FUNCTION 
def predict(text, threshold=0.5):
    text = clean_text(text)
    vec = vectorizer.transform([text]).toarray()
    x = torch.tensor(vec, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0]

    result = {
        LABELS[i]: float(probs[i])
        for i in range(len(LABELS))
        if probs[i] > threshold
    }

    # UI LOGIC
    if len(result) == 0:
        return {"normal": 1.0}

    return result


# GRADIO UI 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Input Comment"),
    outputs=gr.JSON(label="Detected Toxic Labels"),
    title="Toxic Comment Classification (MLP)🤬",
    description="Multi-label toxic comment detection using a simple MLP model.",
) 
demo.launch(share=True)
