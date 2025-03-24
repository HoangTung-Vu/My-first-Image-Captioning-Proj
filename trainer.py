import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from dataloader import get_loader

class Trainer:
    def __init__(self, 
                 data_root='data/flickr8k/Flicker8k_Dataset',
                 captions_file='data/flickr8k/captions.txt',
                 embed_size=256,
                 hidden_size=256,
                 num_layers=1,
                 learning_rate=3e-4,
                 batch_size=32,
                 num_epochs=10,
                 save_step=1,
                 checkpoint_dir='checkpoints',
                 device=None,
                 freeze_cnn_epochs=3):
        """
        Initialize the Trainer with hyperparameters
        """
        # Hyperparameters
        self.data_root = data_root
        self.captions_file = captions_file
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_step = save_step
        self.checkpoint_dir = checkpoint_dir
        self.freeze_cnn_epochs = freeze_cnn_epochs
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Initialize tensorboard writer
        self.writer = SummaryWriter('runs/image_captioning')
        
        # Initialize model, data loader, etc. to None
        self.model = None
        self.train_loader = None
        self.dataset = None
        self.vocab_size = None
        self.criterion = None
        self.optimizer = None
        
    def load_data(self):
        """
        Load and prepare the dataset
        """
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Load data
        self.train_loader, self.dataset = get_loader(
            root_folder=self.data_root,
            annotation_file=self.captions_file,
            transform=transform,
            batch_size=self.batch_size,
        )
        
        # Set vocabulary size
        self.vocab_size = len(self.dataset.vocab)
        
        return self.train_loader, self.dataset
    
    def initialize_model(self, model_class, **model_kwargs):
        """
        Initialize the model, loss function, and optimizer
        
        Args:
            model_class: The model class to instantiate
            model_kwargs: Additional keyword arguments for the model
        """
        # Initialize model with provided class and arguments
        self.model = model_class(**model_kwargs).to(self.device)
        
        # Initialize loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.vocab.stoi["<PAD>"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        return self.model
    
    def load_checkpoint(self, checkpoint_path=None, model_class=None, **model_kwargs):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model_class: The model class to instantiate if model is not initialized
            model_kwargs: Additional keyword arguments for the model
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth.tar')
            
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize model if not already initialized
            if self.model is None and model_class is not None:
                self.model = model_class(**model_kwargs).to(self.device)
                
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Initialize optimizer if not already initialized
            if self.optimizer is None:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {start_epoch-1}")
            return start_epoch
        else:
            print("No checkpoint found, starting from scratch")
            return 0
    
    def save_checkpoint(self, epoch, loss):
        """
        Save model checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}.pth.tar')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'vocab': self.dataset.vocab,
        }, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth.tar')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'vocab': self.dataset.vocab,
        }, latest_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, start_epoch=0, unfreeze_encoder=True):
        """
        Train the model
        
        Args:
            start_epoch: Starting epoch number
            unfreeze_encoder: Whether to unfreeze the encoder after freeze_cnn_epochs
        """
        print("Starting training...")
        for epoch in range(start_epoch, self.num_epochs):
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            
            # Unfreeze CNN after specified number of epochs if model has encoder attribute
            if unfreeze_encoder and epoch == self.freeze_cnn_epochs and hasattr(self.model, 'encoder'):
                print("Unfreezing encoder layers...")
                if hasattr(self.model.encoder, 'train_CNN'):
                    self.model.encoder.train_CNN = True
                # Reinitialize optimizer with all parameters
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate * 0.1)  # Lower learning rate for fine-tuning
            
            # Set model to train mode
            self.model.train()
            
            # Initialize metrics
            epoch_loss = 0
            num_batches = len(self.train_loader)
            
            # Iterate over batches
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
            for idx, (imgs, captions) in enumerate(progress_bar):
                # Move to device
                imgs = imgs.to(self.device)
                captions = captions.to(self.device)
                
                # Forward pass
                outputs = self.model(imgs, captions[:-1])  # Remove <EOS> token for input
                
                # Calculate loss
                # Reshape outputs and targets for loss calculation
                # outputs shape: (seq_length, batch_size, vocab_size)
                # captions shape: (seq_length, batch_size)
                loss = self.criterion(
                    outputs.reshape(-1, self.vocab_size),
                    captions.reshape(-1)
                )
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
                
                # Log to tensorboard (every 100 batches)
                if idx % 100 == 0:
                    self.writer.add_scalar('Training/BatchLoss', loss.item(), epoch * num_batches + idx)
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Training/EpochLoss', avg_loss, epoch)
            
            # Save model checkpoint
            if (epoch + 1) % self.save_step == 0:
                self.save_checkpoint(epoch, avg_loss)
            
            # Evaluate model on a few examples
            if (epoch + 1) % self.save_step == 0:
                self.evaluate_model()
        
        self.writer.close()
        print("Training complete!")
    
    def evaluate_model(self, num_examples=5):
        """
        Evaluate the model on a few examples and print the results
        """
        self.model.eval()
        
        # Get a few random images from the dataset
        indices = np.random.choice(len(self.dataset), num_examples, replace=False)
        
        print("\nEvaluation examples:")
        for idx in indices:
            img, caption = self.dataset[idx]
            img = img.unsqueeze(0).to(self.device)
            
            # Generate caption
            self.model.eval()
            with torch.no_grad():
                # Try both methods if they exist
                if hasattr(self.model, 'caption_image_greedy'):
                    greedy_caption = self.model.caption_image_greedy(img, self.dataset.vocab)
                    greedy_caption = ' '.join([word for word in greedy_caption 
                                              if word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
                else:
                    greedy_caption = "Greedy caption method not available"
                
                if hasattr(self.model, 'caption_image_beam_search'):
                    beam_caption = self.model.caption_image_beam_search(img, self.dataset.vocab, beam_size=3)
                    beam_caption = ' '.join([word for word in beam_caption 
                                            if word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
                else:
                    beam_caption = "Beam search method not available"
            
            # Convert ground truth caption to words
            true_caption = [self.dataset.vocab.itos[idx.item()] for idx in caption]
            
            # Remove special tokens and join words
            true_caption = ' '.join([word for word in true_caption 
                                    if word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
            
            print(f"Image {idx}:")
            print(f"Ground truth: {true_caption}")
            print(f"Greedy prediction: {greedy_caption}")
            print(f"Beam search prediction: {beam_caption}")
            print()