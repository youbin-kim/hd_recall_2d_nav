import numpy as np
from modules.hd_module import hd_module

hd = hd_module()
hd.train_from_file('data/game_dat_simple.out', threshold=True)


sensor_in = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
for i in range(sensor_in.shape[1]):
    act_out=hd.test_sample(sensor_in[i,:])
    print(act_out)
