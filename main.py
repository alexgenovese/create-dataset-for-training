import requests
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2Model
from datasets import load_dataset


def main():
    # this ensures that the current MacOS version is at least 12.3+
    print("---- M1 GPU Acceleration ---")
    print(torch.backends.mps.is_available())

    dataset = load_dataset("Abrumu/Fashion_controlnet_dataset_V3")

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt = "Question: how many cats are there? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    outputs = model(**inputs)

    print(outputs)


if __name__ == "__main__":
    main()