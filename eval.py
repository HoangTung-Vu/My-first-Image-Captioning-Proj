import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, Optional, Any
import time

from model import CNNtoRNN

def load_checkpoint(checkpoint_path: str) -> Tuple[Any, Any]:
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (model, vocab)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # Load checkpoint with appropriate device mapping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint['vocab']
    
    # Initialize model
    model = CNNtoRNN(
        embed_size=256,  # These should match the values used during training
        hidden_size=256,
        vocab_size=len(vocab),
        num_layers=1,
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, vocab

def caption_image(
    image_path: str, 
    model: Any, 
    vocab: Any, 
    device: torch.device, 
    beam_search: bool = True,
    beam_size: int = 5
) -> Tuple[str, float]:
    """
    Generate a caption for an image
    
    Args:
        image_path: Path to the image
        model: Model to use for captioning
        vocab: Vocabulary object
        device: Device to run the model on
        beam_search: Whether to use beam search
        beam_size: Beam size for beam search
        
    Returns:
        Tuple of (caption, inference_time)
    """
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
    start_time = time.time()
    
    with torch.no_grad():
        if beam_search:
            caption = model.caption_image_beam_search(image_tensor, vocab, beam_size=beam_size)
        else:
            caption = model.caption_image_greedy(image_tensor, vocab)
    
    inference_time = time.time() - start_time
    
    # Remove special tokens and join words
    caption_text = ' '.join([word for word in caption 
                        if word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
    
    return caption_text, inference_time

def main():
    """Main function for image captioning"""
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--image_path', type=str, required=True, help='path to image')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/latest_checkpoint.pth.tar', 
                        help='path to checkpoint')
    parser.add_argument('--beam_search', action='store_true', help='use beam search for caption generation')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size for beam search')
    parser.add_argument('--device', type=str, default=None, 
                        help='device (cuda or cpu)')
    parser.add_argument('--save_output', action='store_true', help='save output image with caption')
    parser.add_argument('--output_path', type=str, default='output.png', help='path to save output image')
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}...")
    model, vocab = load_checkpoint(args.checkpoint_path)
    
    # Generate caption
    print(f"Generating caption for {args.image_path}...")
    caption, inference_time = caption_image(
        args.image_path, 
        model, 
        vocab, 
        device, 
        args.beam_search,
        args.beam_size
    )
    
    # Display image and caption
    image = Image.open(args.image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(caption)
    plt.axis('off')
    
    # Save or show the image
    if args.save_output:
        plt.savefig(args.output_path, bbox_inches='tight')
        print(f"Output saved to {args.output_path}")
    else:
        plt.show()
    
    print(f"Caption: {caption}")
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Print additional information
    method = "beam search" if args.beam_search else "greedy search"
    print(f"Method: {method}")
    if args.beam_search:
        print(f"Beam size: {args.beam_size}")
    print(f"Device: {device}")

if __name__ == "__main__":
    main()