import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class OnlineDecisionTransformer(nn.Module):
    """
    Updated Online Decision Transformer for multiband LEO satellite handover.
    Now handles 18-dimensional state space (15 original + 3 band2 metrics).
    """
    
    def __init__(self, state_dim, action_dim, max_length=20, embed_dim=128, 
                 num_layers=3, num_heads=4, dropout=0.1, target_return=1.0):
        super().__init__()
        
        self.state_dim = state_dim  # Now 18 for multiband
        self.action_dim = action_dim
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.target_return = target_return
        
        # Embedding layers for multiband states
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate encoders for band-specific features (optional enhancement)
        self.band1_encoder = nn.Sequential(
            nn.Linear(3, embed_dim // 2),  # SNR, load, capacity for band1
            nn.ReLU()
        )
        
        self.band2_encoder = nn.Sequential(
            nn.Linear(3, embed_dim // 2),  # SNR, load, capacity for band2
            nn.ReLU()
        )
        
        # Return and action embeddings
        self.return_encoder = nn.Linear(1, embed_dim)
        self.action_encoder = nn.Embedding(action_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_length * 3, embed_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head with multiband awareness
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, action_dim)
        )
        
        # Band selection head (helps choose which band to prioritize)
        self.band_selection_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Output: [band1_weight, band2_weight]
            nn.Softmax(dim=-1)
        )
        
        # Experience replay buffer for online learning
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def extract_band_features(self, states):
        """Extract band-specific features from states"""
        # Band1 features: indices 3, 4, 14 (snr, load, capacity)
        band1_features = torch.stack([states[:, :, 3], states[:, :, 4], states[:, :, 14]], dim=-1)
        
        # Band2 features: indices 15, 16, 17 (snr, load, capacity)
        band2_features = torch.stack([states[:, :, 15], states[:, :, 16], states[:, :, 17]], dim=-1)
        
        return band1_features, band2_features
    
    def forward(self, states, actions, returns_to_go, attention_mask=None):
        """
        Forward pass with multiband support
        
        Args:
            states: (batch, seq_len, 18) - Multiband state observations
            actions: (batch, seq_len) - Action indices
            returns_to_go: (batch, seq_len) - Target returns
            attention_mask: (batch, seq_len * 3) - Mask for padding
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Encode states with multiband awareness
        state_embeds = self.state_encoder(states)  # (batch, seq_len, embed_dim)
        
        # Optional: Extract and encode band-specific features
        band1_features, band2_features = self.extract_band_features(states)
        band1_embeds = self.band1_encoder(band1_features)  # (batch, seq_len, embed_dim//2)
        band2_embeds = self.band2_encoder(band2_features)  # (batch, seq_len, embed_dim//2)
        
        # Combine band embeddings (optional enhancement)
        band_embeds = torch.cat([band1_embeds, band2_embeds], dim=-1)  # (batch, seq_len, embed_dim)
        state_embeds = state_embeds + 0.1 * band_embeds  # Weighted combination
        
        # Encode returns and actions
        return_embeds = self.return_encoder(returns_to_go.unsqueeze(-1))  # (batch, seq_len, embed_dim)
        action_embeds = self.action_encoder(actions)  # (batch, seq_len, embed_dim)
        
        # Interleave: (return, state, action) for each timestep
        # Shape: (batch, seq_len * 3, embed_dim)
        sequence = torch.zeros(batch_size, seq_len * 3, self.embed_dim, device=states.device)
        sequence[:, 0::3, :] = return_embeds
        sequence[:, 1::3, :] = state_embeds
        sequence[:, 2::3, :] = action_embeds
        
        # Add positional encoding
        sequence = sequence + self.pos_encoder[:, :seq_len * 3, :]
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to transformer format
            attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
            attention_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
        
        transformer_output = self.transformer(sequence, mask=attention_mask)
        
        # Extract state outputs (every 3rd element starting from index 1)
        state_outputs = transformer_output[:, 1::3, :]  # (batch, seq_len, embed_dim)
        
        # Predict actions
        action_logits = self.action_head(state_outputs)  # (batch, seq_len, action_dim)
        
        # Predict band selection weights
        band_weights = self.band_selection_head(state_outputs)  # (batch, seq_len, 2)
        
        return action_logits, band_weights
    
    def get_action(self, states, actions, returns_to_go, action_mask=None):
        """
        Get action for current state with multiband consideration
        
        Args:
            states: (seq_len, 18) - Sequence of multiband states
            actions: (seq_len,) - Sequence of actions
            returns_to_go: (seq_len,) - Sequence of returns-to-go
            action_mask: (action_dim,) - Boolean mask for valid actions
        """
        # Add batch dimension
        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)
        returns_to_go = returns_to_go.unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            action_logits, band_weights = self.forward(states, actions, returns_to_go)
        
        # Get last timestep
        action_logits = action_logits[0, -1, :]  # (action_dim,)
        band_weights = band_weights[0, -1, :]    # (2,)
        
        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        # Choose action (greedy or sample based on band weights)
        action_probs = F.softmax(action_logits, dim=-1)
        action = torch.argmax(action_probs)
        
        return action.item(), band_weights.cpu().numpy()
    
    def step(self, state, action, reward, next_state, done):
        """
        Online learning step - store experience and update model
        
        Args:
            state: (18,) - Current multiband state
            action: int - Action taken
            reward: float - Reward received
            next_state: (18,) - Next multiband state
            done: bool - Episode termination flag
        """
        # Store experience
        self.memory.append((state, action, reward, next_state, done))
        
        # Update model if enough experiences collected
        if len(self.memory) >= self.batch_size:
            self._update_model()
    
    def _update_model(self):
        """Update model using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare sequences for transformer
        # For online learning, we use shorter sequences
        seq_len = min(5, self.max_length)
        
        # Create training batch
        # Note: This is a simplified update - you may want to implement
        # more sophisticated sequence construction from the replay buffer
        
        states = torch.FloatTensor([exp[0] for exp in batch])  # (batch, 18)
        actions = torch.LongTensor([exp[1] for exp in batch])  # (batch,)
        rewards = torch.FloatTensor([exp[2] for exp in batch])  # (batch,)
        
        # For simplicity, create single-step sequences
        states = states.unsqueeze(1)  # (batch, 1, 18)
        actions = actions.unsqueeze(1)  # (batch, 1)
        returns = rewards.unsqueeze(1)  # (batch, 1)
        
        # Forward pass
        action_logits, _ = self.forward(states, actions, returns)
        
        # Compute loss (simple cross-entropy for now)
        # You may want to implement more sophisticated loss functions
        action_logits = action_logits.squeeze(1)  # (batch, action_dim)
        loss = F.cross_entropy(action_logits, actions.squeeze(1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def set_optimizer(self, optimizer):
        """Set optimizer for online learning"""
        self.optimizer = optimizer
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'embed_dim': self.embed_dim,
            'max_length': self.max_length,
            'target_return': self.target_return
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.target_return = checkpoint['target_return']
        print(f"Model loaded from {path}")
        print(f"State dim: {checkpoint['state_dim']}, Action dim: {checkpoint['action_dim']}")
        print(f"Multiband configuration: 18D state space (15 base + 3 band2 metrics)")


# Helper function for multiband ODT prediction
def predict_multiband_action(model, obs, action_mask, context_length=20):
    """
    Predict action using multiband ODT model
    
    Args:
        model: OnlineDecisionTransformer instance
        obs: (18,) - Current multiband observation
        action_mask: (action_dim,) - Boolean mask for valid actions
        context_length: int - Length of context to use
    """
    # Convert to tensors
    state_tensor = torch.FloatTensor(obs)
    mask_tensor = torch.BoolTensor(action_mask)
    
    # Create dummy context (in practice, maintain trajectory history)
    states = state_tensor.unsqueeze(0)  # (1, 18)
    actions = torch.zeros(1, dtype=torch.long)  # Dummy action
    returns_to_go = torch.FloatTensor([model.target_return])
    
    # Get action and band weights
    action, band_weights = model.get_action(states, actions, returns_to_go, mask_tensor)
    
    return action, band_weights