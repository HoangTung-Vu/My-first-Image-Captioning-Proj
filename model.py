import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(Encoder, self).__init__()
        self.train_CNN = train_CNN # Flag for training CNN, only train the top fc layer
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.resnet(images)

        for name, param in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
        return self.dropout(self.bn(self.relu(features)))

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # Embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        """Caption here is a tensor of shape (batch_size, caption_length) a target caption """
        """Feature is a tensor of shape (batch_size, embed_size) a feature vector from the encoder """
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, trainCNN=False):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(CNNtoRNN, self).__init__()
        self.encoder = Encoder(embed_size, trainCNN)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image_greedy(self, image, vocabulary, max_length=50):
        """Generate a caption from the image. Use when evaluating the model"""
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens).squeeze(0)
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
        
    def caption_image_beam_search(self, image, vocabulary, beam_size=3, max_length=50):
        """Generate a caption from the image using beam search."""
        # Initialize
        with torch.no_grad():
            features = self.encoder(image).unsqueeze(0)
            
            # We'll treat the problem as having a batch size of k
            # Start with just the start token
            start_token = vocabulary.stoi["<SOS>"]
            end_token = vocabulary.stoi["<EOS>"]
            
            # Top k sequences, (score, sequence, hidden_state, cell_state)
            sequences = [(0.0, [start_token], None, None)]
            
            # Loop until max_length or all beams have ended
            for _ in range(max_length):
                all_candidates = []
                
                # Expand each current candidate
                for score, seq, h, c in sequences:
                    # If we've reached the end token, keep this sequence as is
                    if seq[-1] == end_token:
                        all_candidates.append((score, seq, h, c))
                        continue
                    
                    # Feed the last token of the sequence to the model
                    x = self.decoder.embed(torch.LongTensor([seq[-1]]).to(image.device)).unsqueeze(0)
                    
                    # For the first token, use the image features
                    if len(seq) == 1:
                        states = None
                        x = features
                    else:
                        states = (h, c)
                    
                    # Get the next output and states
                    hiddens, states = self.decoder.lstm(x, states)
                    h, c = states
                    output = self.decoder.linear(hiddens.squeeze(0))
                    
                    # Get the top k words
                    log_probs, indices = torch.topk(torch.log_softmax(output, dim=1), beam_size)
                    
                    # Create new candidates
                    for i in range(beam_size):
                        token = indices[0][i].item()
                        log_prob = log_probs[0][i].item()
                        new_score = score + log_prob
                        new_seq = seq + [token]
                        all_candidates.append((new_score, new_seq, h, c))
                
                # Select k best candidates
                sequences = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
                
                # Check if all sequences have ended
                if all(s[1][-1] == end_token for s in sequences):
                    break
            
            # Return the highest scoring sequence
            best_seq = max(sequences, key=lambda x: x[0])[1]
            return [vocabulary.itos[idx] for idx in best_seq]
