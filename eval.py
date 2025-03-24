import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

from model import CNNtoRNN

def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    vocab = checkpoint['vocab']
    
    # Initialize model
    model = CNNtoRNN(
        embed_size=256,  # These should match the values used during training
        hidden_size=256,
        vocab_size=len(vocab),
        num_layers=1,
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, vocab

def caption_image(image_path, model, vocab, device, beam_search=True):
    """Generate a caption for an image"""
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate caption
    model.eval()
    with torch.no_grad():
        if beam_search:
            caption = model.caption_image_beam_search(image_tensor, vocab, beam_size=5)
        else:
            caption = model.caption_image_greedy(image_tensor, vocab)
    
    # Remove special tokens and join words
    caption = ' '.join([word for word in caption 
                        if word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
    
    return caption

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='path to image')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/latest_checkpoint.pth.tar', 
                        help='path to checkpoint')
    parser.add_argument('--beam_search', action='store_true', help='use beam search for caption generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='device')
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device)
    model, vocab = load_checkpoint(args.checkpoint_path)
    model = model.to(device)
    
    # Generate caption
    caption = caption_image(args.image_path, model, vocab, device, args.beam_search)
    
    # Display image and caption
    image = Image.open(args.image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(caption)
    plt.axis('off')
    plt.show()
    
    print(f"Caption: {caption}")

if __name__ == "__main__":
    main()