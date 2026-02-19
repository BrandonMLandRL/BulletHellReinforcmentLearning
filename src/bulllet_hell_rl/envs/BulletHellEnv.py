import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled

from typing import Optional, Union

from bulllet_hell_rl.bullethell import * 

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
    Fire at angle: 0, 90, 180, 360 degrees (4 choices, 90° steps).  

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

        #Comparatory values to detect changes in environment easily
        self.player_previous_health = PLAYER_HEALTH_MAX
        self.player_previous_kill_count = 0


        self.previous_time = 0
        self.current_time = 0
        
        self.max_steps = 10000
        self.step_count = 0

        # Define the action space (5 moves × 36 aim angles = 180 combined actions)
        self.action_space = spaces.Dict({
            "move": spaces.Discrete(5),
            "fire_angle": spaces.Box(
                low=0.0,
                high=270,
                shape=(1,),
                dtype=np.int16
            )
        })

        #For the observation space scaling
        self.N_enemies = 5 #We will track 5 enemies max
        self.N_bullets = 10 #We will track 10 bullets max

        #Define the observation space
        self.observation_space = self.observation_space = spaces.Dict({
            "player": spaces.Box(
                #The following is with normalization
                #Player health from 0 - 1, player x (0,1) player y (0,1)
                low =np.array([0, 0, 0], dtype=np.float32), 
                high=np.array([1, 1.0, 1.0], dtype=np.float32),
                shape=(3,),
                dtype=np.float32
            ),     
            "enemies": spaces.Box(
                #Enemy relative positions go from (-1,1) for x and y
                #Enemy relative velocities also go from (-1,1)
                low=np.tile(np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                            (self.N_enemies, 1)),
                high=np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                            (self.N_enemies, 1)),
                shape=(self.N_enemies, 4),
                dtype=np.float32
            ),
            "bullets": spaces.Box(
                #Bullet relative position and velocity values all range from -1,1
                 low=np.tile(np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                        (self.N_bullets, 1)),
                high=np.tile(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                        (self.N_bullets, 1)),
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
        
        current_time = self.current_time + (60)

        self.current_time = current_time
        delta_time = (current_time - self.previous_time) / 1000

        self.previous_time = current_time
        
         # Spawn enemies at random intervals
        if current_time - self.last_enemy_spawn_time >= self.next_spawn_interval:
            # Spawn enemy at random position
            enemy_x = random.randint(0, WORLD_WIDTH - ENTITY_SIZE)
            enemy_y = random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
            self.enemies.append(Enemy(enemy_x, enemy_y))
            self.last_enemy_spawn_time = current_time
            self.next_spawn_interval = random.randint(ENEMY_SPAWN_MIN, ENEMY_SPAWN_MAX)


        #Get actions for the player
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
        bullets_to_remove = self.player.update(delta_time, None, self.bullets, dir)
        #Set the player's aim angle
        self.player.aim_angle = angle
        for bullet in bullets_to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
        
        #Make the player shoot
        player_bullet = self.player.shoot(current_time)
        if player_bullet:
            self.bullets.append(player_bullet)

        #Update enemies and apply their actions
        for enemy in self.enemies[:]:
            # Enemy shooting
            enemy_bullet = enemy.shoot(current_time)
            if enemy_bullet:
                self.bullets.append(enemy_bullet)
            
            # Enemy update
            bullets_to_remove = enemy.update(delta_time, self.player, current_time, self.bullets)
            for bullet in bullets_to_remove:
                if bullet in self.bullets:
                    self.bullets.remove(bullet)
            
            # Remove dead enemies
            if enemy.health <= 0:
                self.enemies.remove(enemy)

        #Update bullets 
        for bullet in self.bullets[:]:
            bullet.update(delta_time)
            if bullet.is_off_screen():
                self.bullets.remove(bullet)

        #TODO: Update the observation state
        ## Collect the state of the environment. 
        #Collect the N_bullets closest bullets
        closest_bullets = self.get_closest_entities(self.player.x, self.player.y, self.bullets, self.N_bullets)
        #Remove any friendly bullets from the list
        closest_bullets = [b for b in closest_bullets if not b.is_friendly]
        #Pass the bullets into a np.array
        #Create a padded np array for the bullets
        self.bullet_obs = np.zeros((self.N_bullets, 4), dtype=np.float32)
        for i, b in enumerate(closest_bullets[:self.N_bullets]):
            #Create relative coordinates and grab velocities with normalization
            self.bullet_obs[i] = np.array(
                [
                    (b.x - self.player.x)/WORLD_WIDTH,
                    (b.y - self.player.y)/WORLD_HEIGHT,
                    b.vel_x / BULLET_SPEED_ENEMY,
                    b.vel_y / BULLET_SPEED_ENEMY
                ],
            dtype=np.float32
            )
        
        
        #Collect the N_enemies closest enemies
        closest_enemies = self.get_closest_entities(self.player.x, self.player.y, self.enemies, self.N_enemies)

        #Create a padded np array for the enemies
        self.enemies_obs = np.zeros((self.N_enemies, 4), dtype=np.float32)
        for i, enemy in enumerate(closest_enemies[:self.N_enemies]):
            #Create relative to the player cooridnates by subtracting x from p.x and y from p.y
            self.enemies_obs[i] = np.array(
                [
                    (enemy.x - self.player.x)/WORLD_WIDTH,
                    (enemy.y - self.player.y)/WORLD_HEIGHT,
                    enemy.vx,
                    enemy.vy
                ], 
            dtype=np.float32
            ) 
 

        self.state = {
                "player"  : np.array((self.player.health/PLAYER_HEALTH_MAX, self.player.x/WORLD_WIDTH, self.player.y/WORLD_WIDTH)),
                "enemies" : self.enemies_obs,
                "bullets" : self.bullet_obs,
        }

        def is_player_in_center(x,y):
            #Center is center of world
            c_x = WORLD_WIDTH/2
            c_y = WORLD_HEIGHT/2
            #Check if player is in halfworld radius center
            if x > c_x/2 and x < c_x/2 + c_x:
                if y > c_y/2 and y < c_y/2 + c_y:
                    return True
            return False
        reward = 0
        #Supply positive reward for killing enemies
        for event in pygame.event.get():
            if event.type == ENEMY_KILLED:
                self.player.kill_count += 1
        if self.player.kill_count > self.player_previous_kill_count: 
            reward = 50
            self.player_previous_kill_count = self.player.kill_count
            # print(f"we dun got em {self.player.kill_count} ++++100")
            


        #Supply negative reward for getting hit
        elif self.player.health < self.player_previous_health:
            reward = -100
            self.player_previous_health = self.player.health
            # print("took damage -100")
        
        #Supply small positive reward for playing near the world center
        #Specifically if it is WorldWidth/4 radius from world center which is (worldwidth/2,worldheight/2)
       
        elif is_player_in_center(self.player.x, self.player.y):
            reward = 10
        #Supply positive reward for not getting hit
        else:
            reward = 1
        


        #Determine if the game is over 
        terminated = False
        truncated = False
        if self.player.health <= 0:
            terminated = True

        #Truncate the episode if the step count exceeds the max
        if self.step_count >= self.max_steps:
            truncated = True


        self.step_count += 1

        # if self.step_count % 100 == 0:
        #     for i in range(0,len(closest_bullets)):
        #         print(f"Bullet obs: {closest_bullets[i].vel_x}, {closest_bullets[i].vel_y}, isfriendly: {closest_bullets[i].is_friendly}")
        #     print(f"State {self.state}")
        if self.render_mode == "human":
            self.render()


        # Tick the game clock 
        # dt = pygame.time.Clock().tick(60)

        # Build Gymnasium-style info dict
        info = {
            "dt": delta_time,
            "current_time": current_time,
            "step_count": self.step_count,
            "player_health": self.player.health,
            "kill_count": self.player.kill_count,
        }

        # Return observations, reward, terminated, truncated, and info
        return self.state, reward, terminated, truncated, info

    def get_closest_entities(self, ref_x, ref_y, entity_list, num_of_entities):
        if len(entity_list) == 0:
            return []

        # Sort by squared distance (faster, same ordering)
        sorted_entities = sorted(
            entity_list,
            key=lambda e: (e.x - ref_x) ** 2 + (e.y - ref_y) ** 2
        )

        return sorted_entities[:num_of_entities]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.

        Gymnasium API requires reset() to return (observation, info).
        """
        super().reset(seed=seed)

        #Lets set the initial state of the world here 
        self.step_count = 0

        self.previous_time = 0
        self.current_time = 0

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
        
        for i in range(0,3):
            enemy_x = random.randint(0, WORLD_WIDTH - ENTITY_SIZE)
            enemy_y = random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
            e = Enemy(enemy_x, enemy_y)
            self.enemies.append(e)
            # print(f"enemy i: {e.x} , {e.y}")
        
        #Get the enemies that we will use for observations
        enemy_obs_list = self.get_closest_entities(self.player.x, self.player.y, self.enemies, self.N_enemies)

        #Convert the enemy list into np format
        self.enemies_obs = np.zeros((self.N_enemies, 4), dtype=np.float32)
        for i, enemy in enumerate(enemy_obs_list[:self.N_enemies]):
            #Create relative to the player cooridnates by subtracting x from p.x and y from p.y
            self.enemies_obs[i] = np.array(
                [
                    (enemy.x - self.player.x)/WORLD_WIDTH,
                    (enemy.y - self.player.y)/WORLD_HEIGHT,
                    enemy.vx,
                    enemy.vy
                ],
            dtype=np.float32
            )

        self.running = True

        bullets = np.zeros((self.N_bullets, 4), dtype=np.float32)

        # Define state
        self.state = {
            "player": np.array(
                (
                    self.player.health / PLAYER_HEALTH_MAX,
                    self.player.x / WORLD_WIDTH,
                    self.player.y / WORLD_HEIGHT,
                )
            ),
            "enemies": self.enemies_obs,
            "bullets": bullets,
        }

        # Return initial observation and empty info dict (Gymnasium API)
        return self.state, {}

    def render(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Bullet Hell Environment")

        camera_x = self.player.x + self.player.size // 2 - SCREEN_WIDTH // 2
        camera_y = self.player.y + self.player.size // 2 - SCREEN_HEIGHT // 2
        #Clamp camera to world bounds
        camera_x = max(0, min(self.screen_width - self.screen_width, camera_x))
        camera_y = max(0, min(self.screen_height - self.screen_height, camera_y))
        
        #Draw player
        self.player.draw(self.screen, camera_x, camera_y)
        
        #Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen, camera_x, camera_y)
        
        #Draw bullets
        for bullet in self.bullets:
            bullet.draw(self.screen, camera_x, camera_y)

        pygame.display.flip()
    def close(self):
        if self.screen is not None:

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":
    env = BulletHellEnv()
    for i in range(0,10):
        obs, info = env.reset()
        while True:
            # obs_space = env.observation_space
            # print(obs_space)
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                break