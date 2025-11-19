import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99

class QNet(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size).to(device)
        
        self.linear2 = nn.Linear(hidden_size, hidden_size//2).to(device)
        self.value_stream = nn.Linear(hidden_size//2, 1).to(device)
        self.advantage_stream = nn.Linear(hidden_size//2, output_size).to(device)
        

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + advantage - advantage.mean(dim=1, keepdim=True)
        

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name, map_location=device))


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.target = model
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, buffer, frame_idx, batch_size, gamma=GAMMA, beta_start=0.4, beta_frames=600000):
        if len(buffer) < batch_size:
            return

        if frame_idx == beta_frames:
            self.lr /= 10 
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
        transitions, idxs, weights = buffer.sample(batch_size, beta)

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.stack(states)
        
        
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones).unsqueeze(1).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target_q = rewards + gamma * next_q * (~dones)

        td_errors = target_q - q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

        buffer.update_priorities(idxs, td_errors.detach().squeeze().cpu().numpy())
        
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())