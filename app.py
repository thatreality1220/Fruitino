import gradio as gr
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
from private_values import testing_root 
import torchvision


device = torch.device("mps")

fruit_labels = datasets.ImageFolder(root=testing_root).classes

model = torch.load("fruit_modelV2.pt", weights_only=False).to(device)
weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
transform = transforms.Compose([
    transforms.ToTensor()
])
model.eval()

def predict(input):
    input = transform(input)
    img = preprocess(input).to(device)
    print(img)
    if img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    
    with torch.no_grad():
        model_outputs = model(img)
        probabilities = F.softmax(model_outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=-1)
        confidence, prediction = confidence.item(), fruit_labels[prediction.item()]
    return f"Model's Prediction: {prediction}, Confidence: {confidence:.2%}"



app = gr.Interface(fn=predict, inputs=gr.Image(type="numpy"), outputs=["text"])
app.launch(share=True)