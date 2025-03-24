import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
from model import CNNtoRNN

from dataloader import get_loader

class Evaluator:
    def __init__(self,
                 data_root='data/flickr8k/Flicker8k_Dataset',
                 captions_file='data/flickr8k/captions.txt',
                 checkpoint_path='checkpoints/latest_checkpoint.pth.tar',
                 beam_search=True,
                 device=None,
                 batch_size=1):
        """
        Initialize the Evaluator with parameters
        """
        # Parameters
        self.data_root = data_root
        self.captions_file = captions_file
        self.checkpoint_path = checkpoint_path
        self.beam_search = beam_search
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Download NLTK resources
        self.download_nltk_resources()
        
        # Initialize model, data loader, etc. to None
        self.model = None
        self.vocab = None
        self.data_loader = None
        
    def download_nltk_resources(self):
        """Download required NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def load_model(self, model_class=None, checkpoint_path=None, **model_kwargs):
        """
        Load model from checkpoint
        
        Args:
            model_class: The model class to instantiate
            checkpoint_path: Path to the checkpoint file (uses self.checkpoint_path if None)
            model_kwargs: Additional keyword arguments for the model
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        
        # Initialize model with provided class and arguments
        if model_class is not None:
            # If vocab_size is not provided, use the one from the checkpoint
            if 'vocab_size' not in model_kwargs and hasattr(self.vocab, '__len__'):
                model_kwargs['vocab_size'] = len(self.vocab)
                
            self.model = model_class(**model_kwargs).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If no model class is provided, assume the model is already initialized
            if self.model is None:
                raise ValueError("Model must be initialized or model_class must be provided")
                
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        return self.model, self.vocab
    
    def load_data(self):
        """Load and prepare the dataset"""
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Load data
        self.data_loader, _ = get_loader(
            root_folder=self.data_root,
            annotation_file=self.captions_file,
            transform=transform,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle for evaluation
        )
        
        return self.data_loader
    
    def evaluate_bleu(self, max_samples=500):
        """
        Evaluate model using BLEU score
        
        Args:
            max_samples: Maximum number of samples to evaluate
        """
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")
            
        self.model.eval()
        
        # Lists to store references and hypotheses
        references = []
        hypotheses = []
        
        # Process each image in the dataset
        print("Generating captions for evaluation...")
        for idx, (image, caption) in enumerate(tqdm(self.data_loader.dataset)):
            # Prepare image
            image = image.unsqueeze(0).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                if self.beam_search and hasattr(self.model, 'caption_image_beam_search'):
                    predicted_caption = self.model.caption_image_beam_search(image, self.vocab, beam_size=5)
                elif hasattr(self.model, 'caption_image_greedy'):
                    predicted_caption = self.model.caption_image_greedy(image, self.vocab)
                else:
                    raise ValueError("Model must have caption_image_greedy or caption_image_beam_search method")
            
            # Process reference caption
            reference = [self.vocab.itos[idx.item()] for idx in caption]
            reference = [token for token in reference if token not in ["<SOS>", "<PAD>"]]
            # Stop at EOS token
            if "<EOS>" in reference:
                reference = reference[:reference.index("<EOS>")]
            
            # Process predicted caption
            hypothesis = [token for token in predicted_caption if token not in ["<SOS>", "<PAD>"]]
            # Stop at EOS token
            if "<EOS>" in hypothesis:
                hypothesis = hypothesis[:hypothesis.index("<EOS>")]
            
            # Add to lists
            references.append([reference])  # corpus_bleu expects a list of list of references
            hypotheses.append(hypothesis)
            
            # Only process a subset for faster evaluation
            if idx >= max_samples:  # Adjust this number as needed
                break
        
        # Calculate BLEU scores
        print("Calculating BLEU scores...")
        smoothing = SmoothingFunction().method1
        bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        return {
            "BLEU-1": bleu1 * 100,
            "BLEU-2": bleu2 * 100,
            "BLEU-3": bleu3 * 100,
            "BLEU-4": bleu4 * 100
        }
    
    def visualize_examples(self, num_examples=5):
        """
        Visualize some example predictions
        
        Args:
            num_examples: Number of examples to visualize
        """
        if self.model is None:
            raise ValueError("Model must be loaded before visualization")
            
        self.model.eval()
        
        # Get random indices
        indices = np.random.choice(len(self.data_loader.dataset), num_examples, replace=False)
        
        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(indices):
            # Get image and caption
            image, caption = self.data_loader.dataset[idx]
            
            # Generate caption
            image_tensor = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                if hasattr(self.model, 'caption_image_greedy'):
                    greedy_caption = self.model.caption_image_greedy(image_tensor, self.vocab)
                    greedy_caption = ' '.join([token for token in greedy_caption 
                                              if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
                else:
                    greedy_caption = "Greedy caption method not available"
                
                if hasattr(self.model, 'caption_image_beam_search'):
                    beam_caption = self.model.caption_image_beam_search(image_tensor, self.vocab, beam_size=5)
                    beam_caption = ' '.join([token for token in beam_caption 
                                            if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
                else:
                    beam_caption = "Beam search method not available"
            
            # Process reference caption
            reference = [self.vocab.itos[idx.item()] for idx in caption]
            reference = ' '.join([token for token in reference 
                                 if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
            
            # Convert tensor to PIL Image for display
            image_pil = transforms.ToPILImage()(image)
            
            # Plot
            plt.subplot(num_examples, 1, i+1)
            plt.imshow(image_pil)
            plt.title(f"Reference: {reference}\nGreedy: {greedy_caption}\nBeam: {beam_caption}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('evaluation_examples.png')
        plt.close()
        
    def run_evaluation(self, model_class=None, visualize=True, max_samples=500, **model_kwargs):
        """
        Run the full evaluation process
        
        Args:
            model_class: The model class to instantiate
            visualize: Whether to visualize examples
            max_samples: Maximum number of samples to evaluate
            model_kwargs: Additional keyword arguments for the model
        """
        # Load model
        self.load_model(model_class, **model_kwargs)
        
        # Load data
        self.load_data()
        
        # Evaluate BLEU score
        bleu_scores = self.evaluate_bleu(max_samples)
        
        # Print results
        print("\nBLEU Score Evaluation:")
        for metric, score in bleu_scores.items():
            print(f"{metric}: {score:.2f}")
        
        # Save results to file
        with open('bleu_scores.json', 'w') as f:
            json.dump(bleu_scores, f, indent=4)
        
        # Visualize examples if requested
        if visualize:
            print("\nVisualizing examples...")
            self.visualize_examples()
            print("Examples saved to 'evaluation_examples.png'")
            
        return bleu_scores
    

if __name__  == "__main__":
        # Initialize evaluator
    evaluator = Evaluator(
        data_root='data/flickr8k/Flicker8k_Dataset',
        captions_file='data/flickr8k/captions.txt',
        checkpoint_path='checkpoints/latest_checkpoint.pth.tar',
        beam_search=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=1
    )
    
    # Run evaluation with CNNtoRNN model
    evaluator.run_evaluation(
        model_class=CNNtoRNN,
        embed_size=256,
        hidden_size=256,
        num_layers=1,
        visualize=True,
        max_samples=500
    )
