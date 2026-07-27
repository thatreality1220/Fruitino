import gradio as gr
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
from private_values import testing_root 

if torch.mps.is_available:
    device = torch.device("mps")
elif torch.cuda.is_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

fruit_labels = datasets.ImageFolder(root=testing_root).classes

model = torch.load("fruit_model.pt", weights_only=False).to(device)
model.eval()

transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def predict(input):
    img = transform(input).to(device)
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