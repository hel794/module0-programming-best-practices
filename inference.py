
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification

# Load pretrained ResNet-50
model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)
model.eval()

# Load 100 MNIST test images
dataset = load_dataset("ylecun/mnist", split="test[:100]")

correct = 0

for sample in dataset:
    image = sample["image"].convert("RGB").resize((224, 224))
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=-1).item()

    if prediction == sample["label"]:
        correct += 1

accuracy = correct / len(dataset)

print(f"Correct predictions: {correct}/{len(dataset)}")
print(f"Accuracy: {accuracy:.2%}")
