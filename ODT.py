import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from LEOEnvironmentRL import initialize, load_route_from_csv
from HandoverEnvironment import mask_fn
from sb3_contrib.common.wrappers import ActionMasker
import pandas as pd

class DecisionTransformerBlock(nn.Module):
    """Single transformer block for Decision Transformer"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + self.dropout(attn_out))
        
        # MLP with residual connection  
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        
        return x

class DecisionTransformerModel(nn.Module):
    """
    Decision Transformer model for LEO satellite handover decisions.
    Processes sequences of (return-to-go, state, action) triplets.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_length: int = 20,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Embedding layers for each input type
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.action_embedding = nn.Embedding(action_dim, embed_dim)
        self.return_embedding = nn.Linear(1, embed_dim)
        
        # Positional embeddings for each token type in sequence
        # Each timestep has 3 tokens: return-to-go, state, action
        self.pos_embedding = nn.Parameter(torch.zeros(1, 3 * max_length, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DecisionTransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Output heads
        self.action_head = nn.Linear(embed_dim, action_dim)
        self.value_head = nn.Linear(embed_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(
        self,
        returns_to_go: torch.Tensor,  # (batch_size, seq_len, 1)
        states: torch.Tensor,         # (batch_size, seq_len, state_dim)
        actions: torch.Tensor,        # (batch_size, seq_len)
        timesteps: torch.Tensor,      # (batch_size, seq_len)
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embed each input type
        return_embeddings = self.return_embedding(returns_to_go)  # (batch, seq, embed)
        state_embeddings = self.state_embedding(states)           # (batch, seq, embed)
        action_embeddings = self.action_embedding(actions)        # (batch, seq, embed)
        
        # Stack embeddings: [return, state, action, return, state, action, ...]
        # Shape: (batch, seq_len, 3, embed_dim)
        stacked_inputs = torch.stack([
            return_embeddings, 
            state_embeddings, 
            action_embeddings
        ], dim=2)
        
        # Reshape to sequence format: (batch, 3*seq_len, embed_dim)
        stacked_inputs = stacked_inputs.reshape(batch_size, 3*seq_len, self.embed_dim)
        
        # Add positional embeddings
        seq_length = min(stacked_inputs.shape[1], self.pos_embedding.shape[1])
        stacked_inputs = stacked_inputs[:, :seq_length] + self.pos_embedding[:, :seq_length]
        
        # Apply dropout
        x = self.dropout(stacked_inputs)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)
            
        x = self.ln_final(x)
        
        # Extract action and value predictions
        # We want predictions at state positions: indices 1, 4, 7, ... (1 + 3*i)
        action_positions = torch.arange(1, seq_length, 3, device=x.device)
        
        if len(action_positions) > 0:
            action_embeddings_out = x[:, action_positions]  # (batch, seq_len, embed_dim)
            action_logits = self.action_head(action_embeddings_out)  # (batch, seq_len, action_dim)
            values = self.value_head(action_embeddings_out)           # (batch, seq_len, 1)
        else:
            action_logits = torch.zeros(batch_size, 0, self.action_dim, device=x.device)
            values = torch.zeros(batch_size, 0, 1, device=x.device)
        
        return {
            'action_logits': action_logits,
            'values': values.squeeze(-1),
            'embeddings': x
        }

class ExperienceBuffer:
    """Buffer to store trajectory data for online learning"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.trajectories = deque(maxlen=max_size)
        
    def add_trajectory(self, trajectory: Dict):
        """Add a complete trajectory to the buffer"""
        self.trajectories.append(trajectory)
        
    def sample_batch(self, batch_size: int, max_length: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of trajectory segments"""
        if len(self.trajectories) == 0:
            return self._empty_batch(batch_size, max_length)
            
        batch_data = {
            'returns_to_go': [],
            'states': [],
            'actions': [],
            'timesteps': [],
            'attention_mask': []
        }
        
        for _ in range(batch_size):
            if len(self.trajectories) == 0:
                break
                
            # Sample random trajectory
            traj = random.choice(self.trajectories)
            
            # Sample random starting point in trajectory
            traj_len = len(traj['states'])
            if traj_len <= max_length:
                start_idx = 0
                end_idx = traj_len
            else:
                start_idx = random.randint(0, traj_len - max_length)
                end_idx = start_idx + max_length
                
            # Extract trajectory segment
            states = traj['states'][start_idx:end_idx]
            actions = traj['actions'][start_idx:end_idx]
            rewards = traj['rewards'][start_idx:end_idx]
            
            # Calculate returns-to-go
            returns = np.zeros_like(rewards)
            returns[-1] = rewards[-1]
            for i in range(len(rewards) - 2, -1, -1):
                returns[i] = rewards[i] + 0.99 * returns[i + 1]  # Discount factor 0.99
                
            # Pad if necessary
            seq_len = end_idx - start_idx
            if seq_len < max_length:
                pad_len = max_length - seq_len
                states = np.concatenate([states, np.zeros((pad_len, states.shape[1]))])
                actions = np.concatenate([actions, np.zeros(pad_len, dtype=int)])
                returns = np.concatenate([returns, np.zeros(pad_len)])
                
            batch_data['states'].append(states)
            batch_data['actions'].append(actions)
            batch_data['returns_to_go'].append(returns.reshape(-1, 1))
            batch_data['timesteps'].append(np.arange(max_length))
            batch_data['attention_mask'].append(np.ones(max_length, dtype=bool))
            
        # Convert to tensors - FIX: Handle actions separately with correct dtype
        for key in batch_data:
            if len(batch_data[key]) > 0:
                if key == 'actions':
                    # Actions must be long integers for embedding layer
                    batch_data[key] = torch.tensor(np.array(batch_data[key]), dtype=torch.long)
                elif key == 'timesteps':
                    # Timesteps should also be long integers
                    batch_data[key] = torch.tensor(np.array(batch_data[key]), dtype=torch.long)
                elif key == 'attention_mask':
                    # Attention mask should be boolean
                    batch_data[key] = torch.tensor(np.array(batch_data[key]), dtype=torch.bool)
                else:
                    # Everything else (states, returns_to_go) can be float32
                    batch_data[key] = torch.tensor(np.array(batch_data[key]), dtype=torch.float32)
            else:
                batch_data[key] = torch.empty(0)
                
        return batch_data
    
    def _empty_batch(self, batch_size: int, max_length: int) -> Dict[str, torch.Tensor]:
        """Return empty batch when buffer is empty"""
        return {
            'returns_to_go': torch.zeros(batch_size, max_length, 1),
            'states': torch.zeros(batch_size, max_length, 9),  # Assuming 9D state
            'actions': torch.zeros(batch_size, max_length, dtype=torch.long),
            'timesteps': torch.zeros(batch_size, max_length, dtype=torch.long),
            'attention_mask': torch.zeros(batch_size, max_length, dtype=torch.bool)
        }

class OnlineDecisionTransformer:
    """
    Online Decision Transformer agent for LEO satellite handover
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_length: int = 20,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        learning_rate: float = 1e-4,
        target_return: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.max_length = max_length
        self.target_return = target_return
        
        # Initialize model
        self.model = DecisionTransformerModel(
            state_dim=state_dim,
            action_dim=action_dim,
            max_length=max_length,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Current trajectory data
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
        }
        
        # Episode tracking
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
    def predict_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        """Predict action given current state and action mask"""
        self.model.eval()
        
        with torch.no_grad():
            if len(self.episode_states) == 0:
                # First step - create arrays of max_length
                returns_to_go = np.zeros((self.max_length, 1))
                returns_to_go[-1] = self.target_return  # Set target return for last position
                
                states = np.zeros((self.max_length, state.shape[0]))
                states[-1] = state  # Put current state at last position
                
                actions = np.zeros(self.max_length, dtype=int)  # All zeros (dummy actions)
                timesteps = np.arange(self.max_length)
            else:
                # Use recent history
                recent_len = min(len(self.episode_states), self.max_length - 1)
                
                # Get recent trajectory
                recent_states = np.array(self.episode_states[-recent_len:])
                recent_actions = np.array(self.episode_actions[-recent_len:])
                recent_rewards = np.array(self.episode_rewards[-recent_len:])
                
                # Calculate returns-to-go for recent history
                returns = np.zeros(recent_len + 1)
                returns[-1] = self.target_return  # Target for next step
                for i in range(recent_len - 1, -1, -1):
                    returns[i] = recent_rewards[i] + 0.99 * returns[i + 1]
                
                returns_to_go_recent = returns[:-1].reshape(-1, 1)
                
                # Create full-length arrays
                returns_to_go = np.zeros((self.max_length, 1))
                states = np.zeros((self.max_length, state.shape[0]))
                actions = np.zeros(self.max_length, dtype=int)
                
                # Fill in recent history + current state
                history_len = len(recent_states)
                start_idx = self.max_length - history_len - 1
                
                # Place recent history
                returns_to_go[start_idx:start_idx + history_len] = returns_to_go_recent
                states[start_idx:start_idx + history_len] = recent_states
                actions[start_idx:start_idx + history_len] = recent_actions
                
                # Place current state
                returns_to_go[-1] = self.target_return
                states[-1] = state
                actions[-1] = 0  # Dummy action for current step
                
                timesteps = np.arange(self.max_length)
            
            # Convert to tensors
            returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32).unsqueeze(0).to(self.device)
            states = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(0).to(self.device)
            timesteps = torch.tensor(timesteps, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Verify shapes before model call
            assert returns_to_go.shape[1] == self.max_length, f"returns_to_go shape: {returns_to_go.shape}"
            assert states.shape[1] == self.max_length, f"states shape: {states.shape}"
            assert actions.shape[1] == self.max_length, f"actions shape: {actions.shape}"
            assert timesteps.shape[1] == self.max_length, f"timesteps shape: {timesteps.shape}"
            
            # Get predictions
            outputs = self.model(returns_to_go, states, actions, timesteps)
            action_logits = outputs['action_logits']  # (1, seq_len, action_dim)
            
            if action_logits.shape[1] > 0:
                # Get logits for the last (current) timestep
                current_logits = action_logits[0, -1]  # (action_dim,)
                
                # Apply action mask
                masked_logits = current_logits.clone()
                masked_logits[~torch.tensor(mask, dtype=torch.bool, device=self.device)] = -1e10
                
                # Check if any valid actions
                if not mask.any():
                    return -1  # No valid actions
                
                # Sample action
                action_probs = F.softmax(masked_logits, dim=0)
                action = torch.multinomial(action_probs, 1).item()
            else:
                # Fallback: random valid action
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = -1
            
        return action
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Process step and add to current trajectory"""
        # Add to episode tracking
        self.episode_states.append(state.copy())
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        
        if done:
            # End of episode - add trajectory to buffer
            if len(self.episode_states) > 0:
                trajectory = {
                    'states': np.array(self.episode_states),
                    'actions': np.array(self.episode_actions),
                    'rewards': np.array(self.episode_rewards)
                }
                self.buffer.add_trajectory(trajectory)
            
            # Reset episode tracking
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
    
    def train_step(self, batch_size: int = 32):
        """Perform one training step"""
        if len(self.buffer.trajectories) < batch_size:
            return {}  # Not enough data
            
        self.model.train()
        
        # Sample batch
        batch = self.buffer.sample_batch(batch_size, self.max_length)
        
        # Move to device
        for key in batch:
            batch[key] = batch[key].to(self.device)
            
        # Forward pass
        outputs = self.model(
            batch['returns_to_go'],
            batch['states'],
            batch['actions'],
            batch['timesteps']
        )
        
        # Compute loss
        action_logits = outputs['action_logits']  # (batch, seq_len, action_dim)
        target_actions = batch['actions'][:, 1:]  # Next actions (shifted by 1)
        
        # Only compute loss where we have valid targets
        if action_logits.shape[1] > 0 and target_actions.shape[1] > 0:
            min_len = min(action_logits.shape[1], target_actions.shape[1])
            action_logits = action_logits[:, :min_len]
            target_actions = target_actions[:, :min_len]
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                action_logits.reshape(-1, action_logits.shape[-1]),
                target_actions.reshape(-1),
                reduction='mean'
            )
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class LEOEnvDecisionTransformer(gym.Env):
    """
    Wrapper environment for Decision Transformer agent
    """
    
    def __init__(self, constellation_name, route):
        super().__init__()
        
        # Use your existing environment setup
        from HandoverEnvironment import LEOEnv
        self.base_env = LEOEnv(constellation_name, route)
        
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
        # Decision Transformer agent
        self.dt_agent = OnlineDecisionTransformer(
            state_dim=self.observation_space.shape[0],
            action_dim=self.action_space.n,
            max_length=20,
            embed_dim=128,
            num_layers=3,
            target_return=1.0
        )
        
    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        return self.base_env.step(action)
    
    def _get_action_mask(self):
        return self.base_env._get_action_mask()
    
    # Add this property to expose action_mask for the mask_fn
    @property
    def action_mask(self):
        return self.base_env.action_mask
    
    # Forward other attributes that might be needed
    def __getattr__(self, name):
        # If attribute not found in this class, try the base_env
        return getattr(self.base_env, name)

def predict_valid_action_dt(agent, obs, mask):
    """Predict valid action using Decision Transformer"""
    if not np.any(mask):
        print("No valid actions available! Returning penalty action.")
        return -1
    
    action = agent.predict_action(obs, mask)
    return action

def main():
    """Main training loop for Online Decision Transformer"""
    # Setup environment
    inputParams = pd.read_csv("input.csv")
    constellation_name = inputParams['Constellation'][0]
    route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
    
    # Create the base environment first
    base_env = LEOEnvDecisionTransformer(constellation_name, route)
    
    # Then wrap with ActionMasker
    env = ActionMasker(base_env, mask_fn)
    
    # Training parameters
    num_episodes = 100
    train_interval = 5  # Train every 5 episodes
    
    print("Starting Online Decision Transformer training...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        
        while not done:
            # Get action mask - access through the wrapper
            mask = env.action_masks()
            
            # Predict action - access dt_agent through the base environment
            action = predict_valid_action_dt(base_env.dt_agent, obs, mask)
            
            # Take step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Add to agent's experience
            base_env.dt_agent.step(obs, action, reward, next_obs, done or truncated)
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            if done or truncated:
                break
        
        print(f"Episode {episode + 1}: Steps: {step_count}, Reward: {episode_reward:.3f}")
        
        # Train periodically
        if (episode + 1) % train_interval == 0:
            for _ in range(10):  # Multiple training steps
                loss_info = base_env.dt_agent.train_step(batch_size=32)
                if loss_info:
                    print(f"Training loss: {loss_info['loss']:.4f}")
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            base_env.dt_agent.save(f"decision_transformer_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")
    
    # Final save
    base_env.dt_agent.save("decision_transformer_final.pth")
    print("Training completed!")

if __name__ == "__main__":
    main()