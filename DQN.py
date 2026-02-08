#This file will create the DQN that will learn in the BulletHellEnv


#Dict('bullets': Box(-1.0, 1.0, (10, 4), float32), 'enemies': Box(-1.0, 1.0, (5, 4), float32), 'player': Box(0.0, 1.0, (3,), float32))
#Observation space printed: Bullets(N_bullets,4), Enemies (N_enemies,4), Player(3,)

#First we will collect the 


class DeepQNetwork:
    def __init__(self):
        self.obs_dimension = sum(np.prod)

