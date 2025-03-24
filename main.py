import torch
from trainer import Trainer
from model import CNNtoRNN
from evaluate import Evaluator

def main():
    """
    Main entry point for the image captioning model.
    This script uses the Trainer class to train the model.
    """
    # Initialize trainer with hyperparameters
    trainer = Trainer(
        data_root='data/flickr8k/Flicker8k_Dataset',
        captions_file='data/flickr8k/captions.txt',
        embed_size=256,
        hidden_size=256,
        num_layers=1,
        learning_rate=3e-4,
        batch_size=32,
        num_epochs=15,
        save_step=1,
        checkpoint_dir='checkpoints',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        freeze_cnn_epochs=5  # Freeze CNN for first 3 epochs
    )
    
    # Load data
    trainer.load_data()
    
    # Initialize model with CNNtoRNN class
    trainer.initialize_model(
        model_class=CNNtoRNN,
        embed_size=trainer.embed_size,
        hidden_size=trainer.hidden_size,
        vocab_size=trainer.vocab_size,
        num_layers=trainer.num_layers,
        trainCNN=False  # Start with CNN frozen
    )
    
    # Load checkpoint if exists (optional)
    start_epoch = trainer.load_checkpoint(
        model_class=CNNtoRNN,
        embed_size=trainer.embed_size,
        hidden_size=trainer.hidden_size,
        vocab_size=trainer.vocab_size,
        num_layers=trainer.num_layers
    )
    
    # Train model
    trainer.train(start_epoch)

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

if __name__ == "__main__":
    main()