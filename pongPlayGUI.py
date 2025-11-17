import arcade
from envpong import PongGUIEnv
from bot import BotRight, BotLeft
import os
import time
import threading
        
def runLoop(env):
    random_bot = BotRight(env)
    tracker_bot = BotLeft(env)
    
    # simulate for arbitrary 1000 steps (about 33 seconds)
    for i in range(1000):
        actionp1 = tracker_bot.act()
        actionp2 = random_bot.act()
         
        obs, reward, done, truncated, info = env.step(actionp1, actionp2)
        
        tracker_bot.observe(obs)
        random_bot.observe(obs)
        
        # this should be somewhat synced to arcade's event loop
        time.sleep(env.game.dt)
        
def main():
    env = PongGUIEnv()
    
    # Create separate thread for logic updates
    threading.Thread(target=runLoop, args=(env,)).start()
    
    # Arcade thread for GUI updates
    arcade.run()
        
############################
        

if __name__ == "__main__":
    # Set the working directory (where we expect to find files) to the same
    # directory this .py file is in. You can leave this out of your own
    # code, but it is needed to easily run the examples using "python -m"
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    main()