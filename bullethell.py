import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Constants
WORLD_WIDTH = 500
WORLD_HEIGHT = 500
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
ENTITY_SIZE = 20
BULLET_SIZE = 10
PLAYER_SPEED = 5
ENEMY_SPEED = 3
BULLET_SPEED = 2
SHOOT_INTERVAL_ENEMY = 2000  # milliseconds
SHOOT_INTERVAL_PLAYER = 250
BULLET_DAMAGE = 10
ENEMY_SPAWN_MIN = 250  # milliseconds
ENEMY_SPAWN_MAX = 3000  # milliseconds
ENEMY_ACTION_INTERVAL = 250  # milliseconds
PLAYER_HEALTH_MAX = 100

# Colors
BLUE = (0, 100, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class Bullet:
    def __init__(self, x, y, angle, damage, is_friendly):
        self.x = x
        self.y = y
        self.size = BULLET_SIZE
        self.angle = angle  # in degrees
        self.speed = BULLET_SPEED
        self.damage = damage
        self.is_friendly = is_friendly
        # Convert angle to radians and calculate direction vector
        angle_rad = math.radians(angle)
        self.vel_x = math.cos(angle_rad) * self.speed
        self.vel_y = math.sin(angle_rad) * self.speed
        self.rect = pygame.Rect(x, y, self.size, self.size)
    
    def update(self):
        # Update position based on velocity
        self.x += self.vel_x
        self.y += self.vel_y
        self.rect.x = self.x
        self.rect.y = self.y
    
    def is_off_screen(self):
        # Check if bullet is outside world bounds
        return (self.x < 0 or self.x > WORLD_WIDTH or 
                self.y < 0 or self.y > WORLD_HEIGHT)
    
    def draw(self, screen, camera_x, camera_y):
        # Draw bullet relative to camera
        screen_x = self.x - camera_x
        screen_y = self.y - camera_y
        color = BLUE if self.is_friendly else RED
        pygame.draw.circle(screen, color, (int(screen_x + self.size/2), int(screen_y + self.size/2)), self.size//2)

class Entity:
    def __init__(self, x, y, speed, health, is_friendly, shoot_timer_max):
        self.x = x
        self.y = y
        self.size = ENTITY_SIZE
        self.speed = speed
        self.health = health
        self.aim_angle = 0
        self.shoot_timer = 0
        self.shoot_timer_max = shoot_timer_max
        self.is_friendly = is_friendly
        self.action = None  # 'left', 'right', 'up', 'down', or None
        self.rect = pygame.Rect(x, y, self.size, self.size)
    
    def is_colliding(self, bullets):
        """Check if the entity is colliding with any bullets this update.
        If colliding, check if the bullet has the correct friendly relationship to cause damage.
        Returns list of bullets that should be destroyed."""
        bullets_to_remove = []
        for bullet in bullets:
            if self.rect.colliderect(bullet.rect):
                # A friendly bullet can hit the enemy. Unfriendly bullets can hit the player.
                if (self.is_friendly and not bullet.is_friendly) or \
                   (not self.is_friendly and bullet.is_friendly):
                    # Take damage
                    self.health -= bullet.damage
                    if not self.is_friendly:
                        print("hit enemy") 
                    bullets_to_remove.append(bullet)
        return bullets_to_remove
    
    def spawn_bullet(self, damage):
        """Spawns bullet at the AimAngle of the entity.
        Takes in a damage parameter that is applied to the bullet class.
        Uses the aimangle and the entity position to create the direction vector."""
        bullet_x = self.x + self.size // 2
        bullet_y = self.y + self.size // 2
        return Bullet(bullet_x, bullet_y, self.aim_angle, damage, self.is_friendly)
    
    def update_position(self):
        """Update position based on current action."""
        if self.action == 'left':
            self.x -= self.speed
        elif self.action == 'right':
            self.x += self.speed
        elif self.action == 'up':
            self.y -= self.speed
        elif self.action == 'down':
            self.y += self.speed
        
        # Keep entity within world bounds
        self.x = max(0, min(WORLD_WIDTH - self.size, self.x))
        self.y = max(0, min(WORLD_HEIGHT - self.size, self.y))
        
        # Update rect
        self.rect.x = self.x
        self.rect.y = self.y
    
    def draw(self, screen, camera_x, camera_y):
        # Draw entity relative to camera
        screen_x = self.x - camera_x
        screen_y = self.y - camera_y
        color = BLUE if self.is_friendly else RED
        pygame.draw.rect(screen, color, (screen_x, screen_y, self.size, self.size))

class Player(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_SPEED, PLAYER_HEALTH_MAX, True, SHOOT_INTERVAL_PLAYER)
        self.action = None
    
    def update(self, keys, bullets):
        # Handle movement based on keys
        self.action = None
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.action = 'up'
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.action = 'down'
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.action = 'left'
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.action = 'right'
        
        self.update_position()
        
        # Check collisions
        bullets_to_remove = self.is_colliding(bullets)
        return bullets_to_remove
    
    def shoot(self, current_time):
        """Handle shooting at timed intervals with random angle."""
        if current_time - self.shoot_timer >= self.shoot_timer_max:
            # Random angle from 0 to 360 degrees
            self.aim_angle = random.uniform(0, 360)
            self.shoot_timer = current_time
            return self.spawn_bullet(BULLET_DAMAGE)
        return None

class Enemy(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, ENEMY_SPEED, 1, False, SHOOT_INTERVAL_ENEMY)
        self.action = None
        self.last_action_time = 0
        self.action_interval = ENEMY_ACTION_INTERVAL
    
    def update(self, player, current_time, bullets):
        # Choose a new action every action_interval
        if current_time - self.last_action_time >= self.action_interval:
            # Randomly choose a direction
            actions = ['left', 'right', 'up', 'down']
            self.action = random.choice(actions)
            self.last_action_time = current_time
        
        self.update_position()
        
        # Update aim angle to point at player
        dx = player.x + player.size // 2 - (self.x + self.size // 2)
        dy = player.y + player.size // 2 - (self.y + self.size // 2)
        self.aim_angle = math.degrees(math.atan2(dy, dx))
        
        # Check collisions
        bullets_to_remove = self.is_colliding(bullets)
        return bullets_to_remove
    
    def shoot(self, current_time):
        """Handle shooting at timed intervals towards player."""
        if current_time - self.shoot_timer >= self.shoot_timer_max:
            self.shoot_timer = current_time
            return self.spawn_bullet(BULLET_DAMAGE)
        return None

def main():
    # Set up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Bullet Hell Environment")
    clock = pygame.time.Clock()
    
    # Initialize player at random position
    player = Player(
        random.randint(0, WORLD_WIDTH - ENTITY_SIZE),
        random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
    )
    
    # Entity and bullet management
    enemies = []
    bullets = []
    last_shot_time = 0
    last_enemy_spawn_time = 0
    next_spawn_interval = random.randint(ENEMY_SPAWN_MIN, ENEMY_SPAWN_MAX)
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get current time
        current_time = pygame.time.get_ticks()
        
        # Handle player shooting
        keys = pygame.key.get_pressed()
        player_bullet = player.shoot(current_time)
        if player_bullet:
            bullets.append(player_bullet)
        
        # Spawn enemies at random intervals
        if current_time - last_enemy_spawn_time >= next_spawn_interval:
            # Spawn enemy at random position
            enemy_x = random.randint(0, WORLD_WIDTH - ENTITY_SIZE)
            enemy_y = random.randint(0, WORLD_HEIGHT - ENTITY_SIZE)
            enemies.append(Enemy(enemy_x, enemy_y))
            last_enemy_spawn_time = current_time
            next_spawn_interval = random.randint(ENEMY_SPAWN_MIN, ENEMY_SPAWN_MAX)
        
        # Update player
        bullets_to_remove = player.update(keys, bullets)
        for bullet in bullets_to_remove:
            if bullet in bullets:
                bullets.remove(bullet)
        
        # Update enemies
        for enemy in enemies[:]:
            # Enemy shooting
            enemy_bullet = enemy.shoot(current_time)
            if enemy_bullet:
                bullets.append(enemy_bullet)
            
            # Enemy update
            bullets_to_remove = enemy.update(player, current_time, bullets)
            for bullet in bullets_to_remove:
                if bullet in bullets:
                    bullets.remove(bullet)
            
            # Remove dead enemies
            if enemy.health <= 0:
                enemies.remove(enemy)
        
        # Check if player is dead
        if player.health <= 0:
            print("Game Over!")
            running = False
        
        # Update bullets
        for bullet in bullets[:]:
            bullet.update()
            if bullet.is_off_screen():
                bullets.remove(bullet)
        
        # Camera follows player (centered on player)
        camera_x = player.x + player.size // 2 - SCREEN_WIDTH // 2
        camera_y = player.y + player.size // 2 - SCREEN_HEIGHT // 2
        
        # Clamp camera to world bounds
        camera_x = max(0, min(WORLD_WIDTH - SCREEN_WIDTH, camera_x))
        camera_y = max(0, min(WORLD_HEIGHT - SCREEN_HEIGHT, camera_y))
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw player
        player.draw(screen, camera_x, camera_y)
        
        # Draw enemies
        for enemy in enemies:
            enemy.draw(screen, camera_x, camera_y)
        
        # Draw bullets
        for bullet in bullets:
            bullet.draw(screen, camera_x, camera_y)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
