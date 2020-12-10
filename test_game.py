from modules.game_module import game_module
import numpy as np
import matplotlib.pyplot as plt

train_file = './data/game_dat.out'
#train_file = './data/game_dat_test4.out'
game_data = np.loadtxt(train_file, dtype = np.int8, delimiter=',')
n_samples = game_data.shape[0]
print (n_samples)

#threshold = 0.25
#a = game_module()
#a.train_from_file(train_file, threshold)
#success, crash, stuck = a.test_game(1000)

trials = 1000
#%%
thresholds = np.linspace(0.1,0.4,10)
#thresholds = [0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29]
#thresholds = [0.29]
print(thresholds)
success=np.empty((len(thresholds),))
crash=np.empty((len(thresholds),))
stuck=np.empty((len(thresholds),))
num_cond_list = np.empty((len(thresholds),))
num_thrown_list = np.empty((len(thresholds),))
for i,threshold in enumerate(thresholds):
    a = game_module()
    #a.set_sensor_weight(weight)
    a.set_threshold_known(threshold)
    #print(i)
    print(threshold)
    a.train_from_file(train_file)
    num_thrown_list[i] = a.num_thrown
    num_cond_list[i] = a.num_cond
    print(num_thrown_list[i]/n_samples)
    print(num_cond_list[i]/n_samples)
    success[i], crash[i], stuck[i] = a.test_game(trials)
print('done with tests')

#%%




plt.plot(thresholds,success/trials, label='Success')
#plt.plot(thresholds,crash/trials,label='Crash')
#plt.plot(thresholds,stuck/trials,label='Stuck')
plt.plot(thresholds,num_cond_list/n_samples,label='Trained conditions')
plt.plot(thresholds,num_thrown_list/n_samples,label='Rejected conditions')
plt.legend()
plt.title('Success rate and trained condition samples vs. new condition threshold')
plt.xlabel('New condition threshold')
plt.ylabel('Percentage (%)')
plt.show()