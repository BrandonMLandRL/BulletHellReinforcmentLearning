import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import bullethell
#The Env Class accepts two inputs ObsType and ActType.
#Before we implement dual action, we will first get the player to move in the environment. 
#This means only one dimension for actions for now. 
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

    def __init__(render_mode: str | None = "terminal"):
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
        self.enemy_action_interval = 250  # milliseconds
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

        #Define the observation space
        self.observation_space = self.observation_space = spaces.Dict({
            "enemies": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(N_enemies, 4),
                dtype=np.float32
            ),
            "bullets": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(N_bullets, 4),
                dtype=np.float32
            ),
        })
            

    def step(self, action):
        # MOVE_LOOKUP = {
        #     0: (-1, 0),  # Left
        #     1: (1, 0),   # Right
        #     2: (0, 1),   # Up
        #     3: (0, -1),  # Down
        #     4: (0, 0),   # No movement
        # }

        # dx, dy = MOVE_LOOKUP[action["move"]]
        # angle = action["fire_angle"][0]

        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass