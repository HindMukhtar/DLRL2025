import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List
import argparse

# Import ODT components
from ODT import OnlineDecisionTransformer, LEOEnvDecisionTransformer
from HandoverEnvironment import mask_fn
from LEOEnvironmentRL import load_route_from_csv
from sb3_contrib.common.wrappers import ActionMasker


class ODTEvaluator:
    """Evaluates trained Online Decision Transformer models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.agent = None
        self.env = None
        
    def load_model(self):
        """Load the trained ODT model"""
        # Setup environment first to get dimensions
        inputParams = pd.read_csv("input.csv")
        constellation_name = inputParams['Constellation'][0]
        route, route_duration = load_route_from_csv('route.csv', skip_rows=3)
        
        print(f"Setting up environment: {constellation_name}")
        
        # Create environment
        base_env = LEOEnvDecisionTransformer(constellation_name, route)
        self.env = ActionMasker(base_env, mask_fn)
        self.agent = base_env.dt_agent
        
        # Load model weights
        if os.path.exists(self.model_path):
            self.agent.load(self.model_path)
            print(f"ODT model loaded from {self.model_path}")
            
            # Show model performance stats if available
            stats = self.agent.get_performance_stats()
            print(f"Model adaptive target: {stats['current_target']:.3f}")
            if stats['recent_episodes'] > 0:
                print(f"Training episodes in history: {stats['recent_episodes']}")
        else:
            raise FileNotFoundError(f"ODT model file not found: {self.model_path}")
    
    def evaluate_episode(self, target_return: float = None, max_steps: int = 1000) -> Dict:
        """Run a single evaluation episode with ODT"""
        obs, info = self.env.reset()
        done = False
        truncated = False
        step_count = 0
        episode_reward = 0.0
        actions_taken = []
        rewards_received = []
        
        # Set target return (use adaptive target if not specified)
        if target_return is not None:
            self.agent.target_return = target_return
            actual_target = target_return
        else:
            # Use the model's adaptive target
            actual_target = self.agent._get_adaptive_target()
            self.agent.target_return = actual_target
        
        print(f"Evaluating with target return: {actual_target:.3f}")
        
        while not done and not truncated and step_count < max_steps:
            # Get action mask
            mask = self.env.action_masks()
            
            # Predict action
            action = self.agent.predict_action(obs, mask)
            
            if action == -1:  # No valid actions
                print(f"Step {step_count}: No valid actions available!")
                break
            
            # Take step
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Track episode data
            actions_taken.append(action)
            rewards_received.append(reward)
            episode_reward += reward
            step_count += 1
            
            obs = next_obs
        
        return {
            'episode_reward': episode_reward,
            'episode_length': step_count,
            'target_return': actual_target,
            'actions': actions_taken,
            'rewards': rewards_received,
            'success': step_count > 0,
            'target_achievement': episode_reward / max(actual_target, 0.01)
        }
    
    def run_evaluation(self, num_episodes: int = 10, target_returns: List[float] = None) -> Dict:
        """Run multiple evaluation episodes with ODT"""
        # Use adaptive targets by default, or specified targets
        if target_returns is None:
            # Get adaptive target and create variations around it
            adaptive_target = self.agent._get_adaptive_target()
            target_returns = [
                adaptive_target * 0.5,  # Conservative
                adaptive_target,        # Adaptive
                adaptive_target * 1.5,  # Ambitious
                None                    # Let model decide (fully adaptive)
            ]
        
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'target_returns': [],
            'target_achievements': [],
            'success_rates': [],
            'detailed_results': []
        }
        
        print(f"Running ODT evaluation with {num_episodes} episodes per target...")
        print(f"Model's current adaptive target: {self.agent._get_adaptive_target():.3f}")
        
        for i, target_return in enumerate(target_returns):
            target_label = "Adaptive" if target_return is None else f"{target_return:.3f}"
            print(f"\nEvaluating with target: {target_label}")
            
            episode_rewards = []
            episode_lengths = []
            episode_achievements = []
            successes = 0
            
            for episode in range(num_episodes):
                # Run episode
                result = self.evaluate_episode(target_return)
                
                # Collect results
                episode_rewards.append(result['episode_reward'])
                episode_lengths.append(result['episode_length'])
                episode_achievements.append(result['target_achievement'])
                if result['success']:
                    successes += 1
                
                results['detailed_results'].append(result)
                
                print(f"  Episode {episode + 1}/{num_episodes}: "
                      f"Reward = {result['episode_reward']:.3f}, "
                      f"Target = {result['target_return']:.3f}, "
                      f"Achievement = {result['target_achievement']:.2f}")
            
            # Aggregate results for this target return
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_achievement = np.mean(episode_achievements)
            success_rate = successes / num_episodes
            
            results['episode_rewards'].extend(episode_rewards)
            results['episode_lengths'].extend(episode_lengths)
            results['target_returns'].extend([result['target_return'] for result in results['detailed_results'][-num_episodes:]])
            results['target_achievements'].extend(episode_achievements)
            results['success_rates'].append(success_rate)
            
            print(f"  Results for {target_label}:")
            print(f"    Average reward: {avg_reward:.3f}")
            print(f"    Average length: {avg_length:.1f}")
            print(f"    Average achievement ratio: {avg_achievement:.2f}")
            print(f"    Success rate: {success_rate:.2%}")
        
        return results
    
    def plot_evaluation_results(self, results: Dict, save_path: str = "odt_evaluation_results.png"):
        """Plot ODT evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert to numpy arrays for easier plotting
        rewards = np.array(results['episode_rewards'])
        lengths = np.array(results['episode_lengths'])
        targets = np.array(results['target_returns'])
        achievements = np.array(results['target_achievements'])
        
        # 1. Episode Rewards Over Evaluation
        episodes = range(1, len(rewards) + 1)
        axes[0, 0].plot(episodes, rewards, 'bo-', alpha=0.7, markersize=4)
        axes[0, 0].set_title('ODT Episode Rewards During Evaluation')
        axes[0, 0].set_xlabel('Evaluation Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].grid(True)
        
        # 2. Target Achievement Ratios
        axes[0, 1].plot(episodes, achievements, 'go-', alpha=0.7, markersize=4)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Achievement')
        axes[0, 1].set_title('Target Achievement Ratios')
        axes[0, 1].set_xlabel('Evaluation Episode')
        axes[0, 1].set_ylabel('Achieved / Target Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Target vs Achieved Returns Scatter
        axes[1, 0].scatter(targets, rewards, alpha=0.6, c=achievements, cmap='viridis')
        axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', alpha=0.7, label='Perfect Line')
        axes[1, 0].set_title('Target vs Achieved Returns')
        axes[1, 0].set_xlabel('Target Return')
        axes[1, 0].set_ylabel('Achieved Return')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Episode Lengths Distribution
        axes[1, 1].hist(lengths, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Episode Lengths Distribution')
        axes[1, 1].set_xlabel('Episode Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"ODT evaluation plots saved to {save_path}")
    
    def save_results(self, results: Dict, save_path: str = "odt_evaluation_results.npz"):
        """Save ODT evaluation results to file"""
        np.savez(save_path, **results)
        print(f"ODT evaluation results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Online Decision Transformer (ODT)')
    parser.add_argument('--model_path', type=str, default='decision_transformer_final.pth',
                        help='Path to trained ODT model')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes per target return')
    parser.add_argument('--target_returns', type=float, nargs='+', 
                        default=None,
                        help='Specific target returns to evaluate (default: use adaptive targets)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--use_adaptive', action='store_true',
                        help='Use only adaptive targets (ignore --target_returns)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ONLINE DECISION TRANSFORMER (ODT) EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Episodes per target: {args.num_episodes}")
    if args.use_adaptive or args.target_returns is None:
        print("Target returns: Adaptive (model-determined)")
    else:
        print(f"Target returns: {args.target_returns}")
    print(f"Max steps per episode: {args.max_steps}")
    
    try:
        # Create evaluator and load model
        evaluator = ODTEvaluator(args.model_path)
        evaluator.load_model()
        
        # Run evaluation
        target_returns = None if args.use_adaptive else args.target_returns
        results = evaluator.run_evaluation(
            num_episodes=args.num_episodes,
            target_returns=target_returns
        )
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("ODT EVALUATION SUMMARY")
        print("=" * 60)
        
        overall_reward = np.mean(results['episode_rewards'])
        overall_length = np.mean(results['episode_lengths'])
        overall_success = np.mean(results['success_rates'])
        overall_achievement = np.mean(results['target_achievements'])
        
        print(f"Overall average reward: {overall_reward:.3f}")
        print(f"Overall average episode length: {overall_length:.1f}")
        print(f"Overall success rate: {overall_success:.2%}")
        print(f"Overall target achievement ratio: {overall_achievement:.2f}")
        
        print(f"\nBest episode reward: {np.max(results['episode_rewards']):.3f}")
        print(f"Worst episode reward: {np.min(results['episode_rewards']):.3f}")
        print(f"Reward standard deviation: {np.std(results['episode_rewards']):.3f}")
        
        print(f"\nBest target achievement: {np.max(results['target_achievements']):.2f}")
        print(f"Average target return used: {np.mean(results['target_returns']):.3f}")
        
        # Save and plot results
        evaluator.save_results(results)
        evaluator.plot_evaluation_results(results)
        
        print("\nODT evaluation completed successfully!")
        
    except Exception as e:
        print(f"ODT evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()