import torch
from envpong import PongLogic
from model import QNet
from bot_train import get_state, get_action_from_output, get_action_model
import random
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Random bot
class BotRight:
    def __init__(self, env):
        self.env = env
        self.model = QNet(4, 3)
        self.model.load('best_model_final.pth')
        # This bot doesn't require an initial observation
        self.obs = [0]*len(env.observation_space.sample())
        
    def act(self):
        p1_state, p2_state = get_state(self.obs)
        state_tensor = torch.tensor(p2_state, dtype=torch.float32).to(device)
        action = get_action_model(state_tensor, self.model)
        
          
        return get_action_from_output(action)
    
    def observe(self, obs):
        self.obs = obs
        

class BotLeft:
    def __init__(self, env):
        self.env = env
        self.model = QNet(4, 3)
        self.model.load('best_model_final.pth')
        # This bot doesn't require an initial observation
        self.obs = [0]*len(env.observation_space.sample())
    
    def act(self):
        p1_state, p2_state = get_state(self.obs)
        state_tensor = torch.tensor(p1_state, dtype=torch.float32).to(device)
        action = get_action_model(state_tensor, self.model)
        
          
        return get_action_from_output(action)
    
    def observe(self, obs):
        self.obs = obs