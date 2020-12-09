from modules.game_module import game_module
import numpy as np
import matplotlib.pyplot as plt

train_file = './data/game_dat_long.out'

a = game_module()
a.train_from_file(train_file)
success, crash, stuck = a.test_game(1000)


#%%
weights = np.arange(0,10,1)
success=np.empty((len(weights),))
crash=np.empty((len(weights),))
stuck=np.empty((len(weights),))
for i,weight in enumerate(weights):
    a = game_module()
    a.set_sensor_weight(weight)
    a.train_from_file(train_file)
    success[i], crash[i], stuck[i] = a.test_game(1000)

#%%
plt.plot(weights,success, label='Success')
plt.plot(weights,crash,label='Crash')
plt.plot(weights,stuck,label='Stuck')
plt.legend()
plt.show()
plt.xlabel('Sensor data weight')