import pygame
import random
import numpy as np
import os

class game_module:
    def __init__(self):
        self.world_size = (10,10)
        self.grid_size = (self.world_size[0]+2, self.world_size[1]+2)
        self.scale = 50
        self.pixel_dim = (self.grid_size[0]*self.scale, self.grid_size[1]*self.scale)

        self.num_obs = 30


        self.white = (255,255,255)
        self.blue = (0,0,225)
        self.black = (0,0,0)

        self.pos = [0,0]
        self.obs = []
        self.obs_mat = np.zeros(self.world_size)

        self.outdir = './data/'
        self.outfile = self.outdir + 'game_dat.out'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def start_game(self):
        pygame.init()
        screen = pygame.display.set_mode(self.pixel_dim)



        num_block = self.world_size[0]*self.world_size[1]
        obs_idx = random.sample(list(range(num_block)), self.num_obs+1)
        for i in range(self.num_obs):
            row_pos = obs_idx[i]//self.world_size[0]
            col_pos = obs_idx[i]%self.world_size[1]
            self.obs.append((row_pos, col_pos))
            self.obs_mat[row_pos,col_pos] = 1

        self.pos = [obs_idx[-1]//self.world_size[0], obs_idx[-1]%self.world_size[1]]



        running = True

        f = open(self.outfile, 'w')

        while running:
            event = pygame.event.wait()
            screen.fill(self.white)
            self.draw_walls(screen)
            self.draw_obstacles(screen)
            self.draw_me(screen)
            pygame.display.update()
            actuator = 0
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
                f.write(current_sensor + ", " + str(actuator) + "\n")
                print(actuator)
            if (self.check_collision(self.pos[0], self.pos[1])):
                running = False

        pygame.display.quit()
        pygame.quit()
        f.close()

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

    def draw_walls(self, screen):
        pygame.draw.rect(screen, self.black, [0,0,self.pixel_dim[0]-self.scale,self.scale])
        pygame.draw.rect(screen, self.black, [self.pixel_dim[0]-self.scale,0,self.scale,self.pixel_dim[1]-self.scale])
        pygame.draw.rect(screen, self.black, [self.scale,self.pixel_dim[1]-self.scale,self.pixel_dim[0]-self.scale,self.scale])
        pygame.draw.rect(screen, self.black, [0,self.scale,self.scale,self.pixel_dim[1]-self.scale])
        return

    def pos_oob(self, xpos, ypos):
        oob = 0
        if (xpos < 0 or xpos >= self.world_size[0]):
            oob = 1
        if (ypos < 0 or ypos >= self.world_size[1]):
            oob = 1
        return oob

    def check_collision(self, xpos, ypos):
        collision = 0
        if (self.pos_oob(xpos, ypos)):
            collision = 1
        else:
            if (self.obs_mat[xpos, ypos]):
                collision = 1
        return collision

    def get_sensor(self):
        sensor_pos = [(self.pos[0]+1, self.pos[1]),
                (self.pos[0]-1, self.pos[1]),
                (self.pos[0], self.pos[1]+1),
                (self.pos[0], self.pos[1]-1)]
        sensor_act = [self.check_collision(xpos,ypos) for (xpos,ypos) in sensor_pos]
        print(sensor_act)
        return "{}, {}, {}, {}".format(sensor_act[0], sensor_act[1], sensor_act[2], sensor_act[3])
