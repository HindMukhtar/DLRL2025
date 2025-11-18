# import libraries 
import HandoverEnvironment as he
import numpy as np 


env = he.BeamHandoverEnv() 
obs, _ = env.reset()

for _ in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    print(f"Step: {_}, Action: {action}, Observation: {obs}, Reward: {reward}")
    if done:
        print("Episode finished.")
        break   

