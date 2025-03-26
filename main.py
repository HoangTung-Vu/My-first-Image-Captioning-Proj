import torch
import argparse
import os
from typing import Dict, Any
import json
from trainer import Trainer
from model import CNNtoRNN
from evaluate import Evaluator

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Image Captioning')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='data/flickr8k/Flicker8k_Dataset',
                        help='path to dataset images')
    parser.add_argument('--captions_file', type=str, default='data/flickr8k/captions.txt',
                        help='path to captions file')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,  # Increased from 256 to 512
                        help='size of word embeddings')
    parser.add_argument('--hidden_size', type=int, default=512,  # Increased from 256 to 512
                        help='size of LSTM hidden state')
    parser.add_argument('--num_layers', type=int, default=1, 
                        help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--freeze_cnn_epochs', type=int, default=8,  # Reduced to allow earlier fine-tuning
                        help='number of epochs to freeze CNN')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--save_step', type=int, default=2,
                        help='frequency of saving checkpoints (in epochs)')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='use mixed precision training')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='patience for early stopping')
    
    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--beam_search', action='store_true',
                        help='use beam search for evaluation')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='beam size for beam search')
    parser.add_argument('--visualize', type=bool, default=True,
                        help='visualize examples during evaluation')
    parser.add_argument('--visualization_dir', type=str, default='visualization_results',
                        help='directory to save visualization results')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                        help='device to use (cuda or cpu)')
    
    return parser.parse_args()

def save_config(args: argparse.Namespace, config_path: str) -> None:
    """Save configuration to file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to {config_path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main() -> None:
    """
    Main entry point for the image captioning model.
    This script uses the Trainer class to train the model and the Evaluator class to evaluate it.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Save configuration
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    save_config(args, config_path)
    
    # Initialize trainer with hyperparameters
    trainer = Trainer(
        data_root=args.data_root,
        captions_file=args.captions_file,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_step=args.save_step,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        freeze_cnn_epochs=args.freeze_cnn_epochs,
        use_mixed_precision=args.use_mixed_precision,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Load data
    trainer.load_data()
    
    # Initialize model with CNNtoRNN class
    trainer.initialize_model(
        model_class=CNNtoRNN,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=trainer.vocab_size,
        num_layers=args.num_layers,
        trainCNN=False,  # Start with CNN frozen
        dropout_rate=args.dropout
    )
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume:
        checkpoint_path = args.checkpoint_path or os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth.tar')
        start_epoch = trainer.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model_class=CNNtoRNN,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            vocab_size=trainer.vocab_size,
            num_layers=args.num_layers,
            dropout_rate=args.dropout
        )
    
    # Train model if not evaluating
    if not args.evaluate:
        trainer.train(start_epoch=start_epoch)
    
    # Evaluate model
    if args.evaluate or not args.evaluate:  # Always evaluate after training
        print("\nEvaluating model...")
        
        # Initialize evaluator
        evaluator = Evaluator(
            data_root=args.data_root,
            captions_file=args.captions_file,
            checkpoint_path=args.checkpoint_path or os.path.join(args.checkpoint_dir, 'best_model.pth.tar'),
            beam_search=args.beam_search,
            device=device,
            batch_size=args.batch_size,
            visualization_dir=args.visualization_dir
        )
        
        # Run evaluation
        bleu_scores = evaluator.run_evaluation(
            model_class=CNNtoRNN,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            visualize=args.visualize,
            dropout_rate=args.dropout
        )
        
        print("\nFinal BLEU Scores:")
        for metric, score in bleu_scores.items():
            print(f"{metric}: {score:.2f}")

if __name__ == "__main__":
    main()