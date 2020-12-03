import pygame
import random
import numpy as np
import os

from modules.hd_module import hd_module

class game_module:
    def __init__(self):
        self.world_size = (10,10)
        self.grid_size = (self.world_size[0]+2, self.world_size[1]+2)
        self.scale = 50
        self.pixel_dim = (self.grid_size[0]*self.scale, self.grid_size[1]*self.scale)

        self.num_obs = 30

        self.white = (255,255,255)
        self.blue = (0,0,225)
        self.green = (0,255,0)
        self.black = (0,0,0)

        self.pos = [0,0]
        self.goal_pos = [0,0]
        self.obs = []
        self.obs_mat = np.zeros(self.world_size)

        self.hd_module = hd_module() 

        self.outdir = './data/'
        self.outfile = self.outdir + 'game_dat.out'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)


    def setup_game(self):
        num_block = self.world_size[0]*self.world_size[1]
        obs_idx = random.sample(list(range(num_block)), self.num_obs+1)
        for i in range(self.num_obs):
            row_pos = obs_idx[i]//self.world_size[0]
            col_pos = obs_idx[i]%self.world_size[1]
            self.obs.append((row_pos, col_pos))
            self.obs_mat[row_pos,col_pos] = 1

        self.pos = [obs_idx[-1]//self.world_size[0], obs_idx[-1]%self.world_size[1]]
        self.random_goal_location()
        return

    def train_from_file(self, filename):
        self.hd_module.train_from_file(filename)

    def play_game(self, gametype):
        pygame.init()
        screen = pygame.display.set_mode(self.pixel_dim)

        running = True

        f = open(self.outfile, 'w')

        while running:
            self.game_step(gametype, screen)

            pygame.display.update()
            actuator = 0

            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                current_sensor = self.get_sensor()
                if event.key == pygame.K_LEFT:
                    self.pos[0] -= 1
                    actuator = 0
                elif event.key == pygame.K_RIGHT:
                    self.pos[0] += 1
                    actuator = 1
                elif event.key == pygame.K_UP:
                    self.pos[1] -= 1
                    actuator = 2
                elif event.key == pygame.K_DOWN:
                    self.pos[1] += 1
                    actuator = 3
# *********************** CHANGE BASED ON SENSOR DATA *************************
                sensor_str = "{}, {}, {}, {}".format(*current_sensor)
                f.write(sensor_str + ", " + str(actuator) + "\n")
# *****************************************************************************
            if (self.check_collision(self.pos[0], self.pos[1])):
                running = False

        pygame.display.quit()
        pygame.quit()
        f.close()
        return

    def autoplay_game(self, gametype):
        pygame.init()
        screen = pygame.display.set_mode(self.pixel_dim)
        clock = pygame.time.Clock()
        running = True

        while running:
            self.game_step(gametype, screen)

            pygame.display.update()
            actuator = 0

            clock.tick(3)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            current_sensor = self.get_sensor()
            act_out = self.hd_module.test_sample(current_sensor)
            if act_out == 0:
                self.pos[0] -= 1
            elif act_out == 1:
                self.pos[0] += 1
            elif act_out == 2:
                self.pos[1] -= 1
            elif act_out == 3:
                self.pos[1] += 1

            if (self.check_collision(self.pos[0], self.pos[1])):
                running = False

        pygame.display.quit()
        pygame.quit()
        return

    def game_step(self, gametype, screen):
        screen.fill(self.white)
        self.draw_walls(screen)
        self.draw_obstacles(screen)
        self.draw_me(screen)
        if (gametype):
            if self.goal_pos == self.pos:
                self.random_goal_location()
            self.draw_goal(screen)
        return

    def draw_me(self, screen):
        xpixel = (self.pos[0]+1)*self.scale
        ypixel = (self.pos[1]+1)*self.scale
        pygame.draw.rect(screen, self.blue, [xpixel,ypixel,self.scale,self.scale])
        return
        

    def draw_obstacles(self, screen):
        for pos in self.obs:
            xpos = (pos[0]+1)*self.scale
            ypos = (pos[1]+1)*self.scale
            pygame.draw.rect(screen, self.black, [xpos,ypos,self.scale,self.scale])
        return

    def draw_goal(self, screen):
        xpixel = (self.goal_pos[0]+1)*self.scale
        ypixel = (self.goal_pos[1]+1)*self.scale
        pygame.draw.rect(screen, self.green, [xpixel,ypixel,self.scale,self.scale])
        return

    def draw_walls(self, screen):
        pygame.draw.rect(screen, self.black, [0,0,self.pixel_dim[0]-self.scale,self.scale])
        pygame.draw.rect(screen, self.black, [self.pixel_dim[0]-self.scale,0,self.scale,self.pixel_dim[1]-self.scale])
        pygame.draw.rect(screen, self.black, [self.scale,self.pixel_dim[1]-self.scale,self.pixel_dim[0]-self.scale,self.scale])
        pygame.draw.rect(screen, self.black, [0,self.scale,self.scale,self.pixel_dim[1]-self.scale])
        return

    def pos_oob(self, xpos, ypos):
        # Check if (xpos,ypos) is out of bounds
        oob = 0
        if (xpos < 0 or xpos >= self.world_size[0]):
            oob = 1
        if (ypos < 0 or ypos >= self.world_size[1]):
            oob = 1
        return oob

    def check_collision(self, xpos, ypos):
        # Check if (xpos,ypos) is out of bounds or occupied by object
        collision = 0
        if (self.pos_oob(xpos, ypos)):
            collision = 1
        else:
            if (self.obs_mat[xpos, ypos]):
                collision = 1
        return collision

    def random_goal_location(self):
        # Choose random unoccupied square for the goal position
        num_block = self.world_size[0]*self.world_size[1]
        goal_idx = random.randrange(num_block)
        row_pos = goal_idx//self.world_size[0]
        col_pos = goal_idx%self.world_size[1]
        while (self.check_collision(row_pos,col_pos)):
            goal_idx = random.randrange(num_block)
            row_pos = goal_idx//self.world_size[0]
            col_pos = goal_idx%self.world_size[1]
        self.goal_pos = [row_pos, col_pos]
        return
        

# *********************** CHANGE BASED ON SENSOR DATA *************************
    def get_sensor(self):
        # list of coordinates for squares around current position
        sensor_pos = [(self.pos[0]-1, self.pos[1]),
                (self.pos[0]+1, self.pos[1]),
                (self.pos[0], self.pos[1]-1),
                (self.pos[0], self.pos[1]+1)]
        sensor_vals = [self.check_collision(xpos,ypos) for (xpos,ypos) in sensor_pos]
        return sensor_vals
# *****************************************************************************
