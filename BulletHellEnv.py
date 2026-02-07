import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled

from typing import Optional, Union

from bullethell import * 

class BulletHellEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    ## Description 
    The environment is a 500x500 pixel world where enemies will spawn in over time and the player must eliminate the enemies without being eliminated.
    Squares are entities who exist in the world. The player is a blue square entity and the enemies are red square entities.
    All entities can fire bullets. The player must dodge the bullets and hit enemies with their own bullets. 

    ## Action Space

    The action space is a `ndarray` with shape `(3)` which could look like `{1, 0, 180}` where idx 0 is X axis movement, idx 1 is y axis, and idx 2 is angle for shoot.
    The actions available to the player include two simultaneous options: Move in direction and Fire at angle.

    Move in direction: (-1,0) : Left, (1,0) : Right, (0,1) : Up, (0,-1) Down, and (0,0) : No Movement. These are 5 choices so we can use a discrete space of 5
    Fire at angle: 0 - 360 degrees.  

    ## Observation Space

    The observation space is only a segment of what information is actually visible for the agent here. 
    The observation space will consist of the 5 nearest enemy positions + their velocity(vx,vy). 
    Additionally it will consist of the 10 nearest bullet positions and their velocities(vx,vy).
    Finally, it will also contain the player's health and position. for a total of 60 observations

    5*4 + 10 * 4 = 60 + 3 for player 

    *We may need to design further some more careful observations that are more actionable. Rather than direct world observations but fuck it lets try

    ## Rewards

    The goal of the player is to hit enemies without getting hit.

    if player hits enemy:
        +100

    if player gets hit by enemy:
        -100

    else:
        +1 for staying safe. 

    ** wE MIgHT WAnT TO COnsIDER MAKing pLaYER diE IN OnE HIt

    ## Starting state 
    
    Player is assigned a random position within the world space. One enemy is spawned in. Enemies position will be checked for overlap with player when spawning them in to ensure no overlap.

    ## Epsiode End
    
    When the player's health reaches <= 0 
                or
    When the episode length is > 500.

    ## Arguments

    render_mode as a keyword for gymnasium.make(). 
    The user can either choose "human" or "terminal"

    ## Vectorized environment

    ** Could be cool to investigate Looks liek you need to write your very own Vector Environment

    """

    metadata = {
        "render_modes": ["human", "terminal"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str | None = "terminal"):
        self.world_width = 500
        self.world_height = 500
        self.screen_width = 500
        self.screen_height = 500
        self.entity_size = 20
        self.bullet_size = 10
        self.player_speed = 5
        self.enemy_speed = 3
        self.bullet_speed = 2
        self.shoot_interval_enemy = 2000  # milliseconds
        self.shoot_interval_player = 250
        self.bullet_damage = 10
        self.enemy_spawn_min = 250        # milliseconds
        self.enemy_spawn_max = 3000       # milliseconds
        self.enemy_action_interval = 1500  # milliseconds
        self.player_health_max = 100

    
        #Define the actionspace 
        self.action_space = spaces.Dict({
            "move": spaces.Discrete(5),
            "fire_angle": spaces.Box(
                low=0.0,
                high=360.0,
                shape=(1,),
                dtype=np.float32
            )
        })

        #For the observation space scaling
        self.N_enemies = 5 #We will track 5 enemies max
        self.N_bullets = 10 #We will track 10 bullets max

        #Define the observation space
        self.observation_space = self.observation_space = spaces.Dict({
            "player": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),     
            "enemies": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.N_enemies, 4),
                dtype=np.float32
            ),
            "bullets": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.N_bullets, 4),
                dtype=np.float32
            ),
        })
            
        self.state = None
        self.render_mode = render_mode

        # print(f"{render_mode}")

    def step(self, action): 
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # Get current time
        current_time = pygame.time.get_ticks()
        
         # Spawn enemies at random intervals
        if current_time - self.last_enemy_spawn_time >= self.next_spawn_interval:
            # Spawn enemy at random position
            enemy_x = random.randint(0, WORLD_WIDTH - ENTITY_SIZE)
            enemy_y = random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
            self.enemies.append(Enemy(enemy_x, enemy_y))
            self.last_enemy_spawn_time = current_time
            self.next_spawn_interval = random.randint(ENEMY_SPAWN_MIN, ENEMY_SPAWN_MAX)

        MOVE_LOOKUP = {
            0: 'left',  # Left
            1: 'right',   # Right
            2: 'up',   # Up
            3: 'down',  # Down
            4: 'none',   # No movement
        }

        dir = MOVE_LOOKUP[action["move"]]
        angle = action["fire_angle"][0]

        #Update player and apply action
        bullets_to_remove = self.player.update(None, self.bullets, dir)
        self.player.aim_angle = angle
        for bullet in bullets_to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)

        ## Collect the state of the environment. 

        #Get health and position of player
        player_health = self.player.health
        player_global_position = np.array(self.player.x, self.player.y)        

        #List of size self.N_enemies
        enemy_obs = None
        #Max distance and idx of it in enemy_obs
        max_dist_idx = 0
        max_dist_value = 0
        dists = None
        i = 0
        for e in self.enemies:
            #Calculate the relative position to the player.
            vx = e.vx
            vy = e.vy
            rel_x = e.x - self.player.x
            rel_y = e.y - self.player.y
            dist = np.linalg.norm(np.array(e.x,e.y) - player_global_position)
            print(f"Distance to player {dist}")
            if len(enemy_obs) > self.N_enemies - 1:
                #If the enemy is closer than any enemy in the list, replace the enemy with the greatest distance 
                if dist < max_dist_value:
                    #Replace the entry at that index with this enemy
                    enemy_obs[max_dist_idx] = (rel_x,rel_y,vx,vy)
                    dists[max_dist_idx] = dist
                    #Find the new greatest max in the array
                    m = 0 #temp max
                    j = 0 #iterator
                    for i in dists:
                        if i > m:
                            m = i
                            max_dist_idx = j
                        j += 1
                    max_dist_value = m

            else:
                #Add the enemy to the observations and record if it has the greatest distance
                if dist > max_dist_value:
                    max_dist_value = dist
                    max_dist_idx = i
                enemy_obs.append((rel_x,rel_y,vx,vy))
                dists.append(dist)
                i += 1 

        print(enemy_obs)
        
        #Collect the 


        #Return np array of observations, reward, terminated, 
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


    def reset(self, seed: Optional[int] = None,):

        super().reset(seed=seed)

        #Lets set the initial state of the world here 

        #Initialize player at random position
        self.player = Player(
            random.randint(0, WORLD_WIDTH - ENTITY_SIZE),
            random.randint(0, WORLD_HEIGHT - ENTITY_SIZE),
            True
        )

        #Entity and bullet management
        self.enemies = []
        self.bullets = []
        self.last_shot_time = 0
        self.last_enemy_spawn_time = 0
        self.next_spawn_interval = random.randint(ENEMY_SPAWN_MIN, ENEMY_SPAWN_MAX)
        
        for i in range(0,10):
            enemy_x = random.randint(0, WORLD_WIDTH - ENTITY_SIZE)
            enemy_y = random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
            self.enemies.append(Enemy(enemy_x, enemy_y))

        self.running = True

        #Define state
        self.state()

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = BulletHellEnv()
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        env.render()

        if done:
            break