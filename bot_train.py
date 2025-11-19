import copy
import math
import torch
import random
import numpy as np
from collections import deque
from buffer import PrioritizedReplayBuffer
from envpong import PongLogic
from envpong import PongEnv
from model import QNet, QTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_MEMORY = 200_000
BATCH_SIZE = 64
LR = 0.0001
EPSILON_DECAY = 50000
MIN_EPSILON = 0.05

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1 
        self.gamma = 0.99 
        self.memory = PrioritizedReplayBuffer(capacity=MAX_MEMORY, alpha=0.5)
        tam = 4
        self.model = QNet(tam, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done)) 


    def train_memory(self, frame_idx, batch_size=BATCH_SIZE):
        
        self.trainer.train_step(self.memory, frame_idx, batch_size)
        
    

    def get_action(self, state):
       
        final_move = [0,0,0]
        
        if (np.random.uniform() < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            
            prediction = self.model(state.unsqueeze(0))
            move = prediction.argmax().item()
            final_move[move] = 1
            
        
        self.epsilon = max(MIN_EPSILON, self.epsilon * np.exp(-1.0 / EPSILON_DECAY))
            
        

        return final_move
    
def get_action_model(state, model):
    prediction = model(state.unsqueeze(0))
    move = prediction.argmax().item()
    final_move = [0,0,0]
    final_move[move] = 1
    return final_move

def get_state(obs):
    # inputs += [state.paddle1Position[0]/self.game.windowWidth]
    # inputs += [state.paddle1Position[1]/self.game.windowHeight]
    # inputs += [state.paddle1Velocity[0]/(self.game.ballVelocityMag*100)]
    # inputs += [state.paddle1Velocity[1]/(self.game.ballVelocityMag*100)]
    # inputs += [state.paddle2Position[0]/self.game.windowWidth]
    # inputs += [state.paddle2Position[1]/self.game.windowHeight]
    # inputs += [state.paddle2Velocity[0]/(self.game.ballVelocityMag*100)]
    # inputs += [state.paddle2Velocity[1]/(self.game.ballVelocityMag*100)]
    # inputs += [state.ballPosition[0]/self.game.windowWidth]
    # inputs += [state.ballPosition[1]/self.game.windowHeight]
    # inputs += [state.ballVelocity[0]/(self.game.ballVelocityMag*100)]
    # inputs += [state.ballVelocity[1]/(self.game.ballVelocityMag*100)]
    # inputs += [state.player1action]
    # inputs += [state.player2action]
    
    ang = math.degrees(math.atan2(obs[11], obs[10]))/180
    p1 = [obs[1], obs[8], obs[9], ang]
    
    
    ang = math.degrees(math.atan2(obs[11], -obs[10]))/180
    p2 = [obs[5], (1 - obs[8]), obs[9], ang]
    
    return p1, p2

def get_reward(state_new, state_old, aux=0):
    reward_p1 = 0
    reward_p2 = 0
    if state_new[0] >= state_new[8]:
        reward_p1 -= 1
        aux += 1
        aux *= -1
        
    if state_new[4] <= state_new[8]:
        reward_p2 -= 1
        aux += 1
        aux *= -1
        
    if state_new[10] > 0 and state_old[10] < 0:
        reward_p1 += 1 
        aux += 1
        
    if state_new[10] < 0 and state_old[10] > 0:
        reward_p2 += 1 
        aux += 1
        
    return reward_p1, reward_p2, aux
    
def get_action_from_output(output):
    move = np.argmax(output)
    if move == 0:
        return PongLogic.PaddleMove.DOWN
    elif move == 1:
        return PongLogic.PaddleMove.STILL
    else:
        return PongLogic.PaddleMove.UP
    

def train(game):
    record = 0
    agent = Agent()
    
    
    count = 0
    obs, info = game.reset()
    
    p1_state, p2_state = get_state(obs)
    med = 0
    reset_aux = 0
    while True:
        count += 1

        state_old_p1 = torch.tensor(p1_state, dtype=torch.float32).to(device)
        state_old_p2 = torch.tensor(p2_state, dtype=torch.float32).to(device)

        actionp1 = agent.get_action(state_old_p1)
        actionp2 = agent.get_action(state_old_p2)
        
        
        state_old = copy.deepcopy(obs)
        
        actionp1_converted = get_action_from_output(actionp1)
        actionp2_converted = get_action_from_output(actionp2)
        
        obs, reward, done, truncated, info = env.step(actionp1_converted, actionp2_converted)
        p1_state, p2_state = get_state(obs)
        reward_p1, reward_p2, reset_aux = get_reward(obs, state_old, reset_aux)
        
        if reset_aux < 0:
            done = True
        
        state_new_p1 = torch.tensor(p1_state, dtype=torch.float32).to(device)
        state_new_p2 = torch.tensor(p2_state, dtype=torch.float32).to(device)

        agent.remember(state_old_p1, np.argmax(actionp1), reward_p1, state_new_p1, done)
        agent.remember(state_old_p2, np.argmax(actionp2), reward_p2, state_new_p2, done)
        
        if(count > 1000):
            agent.train_memory(count)

        if(count % 1000 == 0):
            agent.trainer.update_target()
            
        if count % 10000 == 0:
            agent.model.save()
            print("Model Saved at count:", count)
            print("Games played:", agent.n_games)
            if agent.n_games != 0:
                print("Median resets:", med/agent.n_games)
            if med > record:
                record = med
                agent.model.save(file_name='best_model.pth')
            agent.n_games = 0
            med = 0
        
            
            
        if count % 100000 == 0:
            agent.model.save('model_at_'+str(count)+'.pth')
            if count >= 1000000:
                break
            
        if reset_aux >= 50 or done:
            obs, info = game.reset()
            p1_state, p2_state = get_state(obs)
            med += abs(reset_aux)
            agent.n_games += 1
            reset_aux = 0        


if __name__ == '__main__':
    env = PongEnv(debugPrint=False)
    train(env)