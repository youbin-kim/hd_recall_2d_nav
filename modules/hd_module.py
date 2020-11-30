import random
import numpy as np
import os

class hd_module:
    def __init__(self):
        # HD dimension used
        self.dim = 10000
        self.num_sensors = 4
        self.num_actuators = 4

        self.outdir = './data/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        sensor_ids_fname = self.outdir + 'hd_sensor_ids_dim_' + str(self.dim) + '.npy'
        sensor_vals_fname = self.outdir + 'hd_sensor_vals_dim_' + str(self.dim) + '.npy'
        actuator_vals_fname = self.outdir + 'hd_actuator_vals_dim_' + str(self.dim) + '.npy'

        # Load/create HD items
        if os.path.exists(sensor_ids_fname):
            self.hd_sensor_ids = np.load(sensor_ids_fname)
        else:
            self.hd_sensor_ids = self.create_bipolar_mem(self.num_sensors,self.dim)
            np.save(sensor_ids_fname, self.hd_sensor_ids)

        if os.path.exists(sensor_vals_fname):
            self.hd_sensor_vals = np.load(sensor_vals_fname)
        else:
            self.hd_sensor_vals = self.create_bipolar_mem(2,self.dim)
            np.save(sensor_vals_fname, self.hd_sensor_vals)

        if os.path.exists(actuator_vals_fname):
            self.hd_actuator_vals = np.load(actuator_vals_fname)
        else:
            self.hd_actuator_vals = self.create_bipolar_mem(self.num_actuators,self.dim)
            np.save(actuator_vals_fname, self.hd_actuator_vals)

        # Initialize program vector
        self.hd_program_vec = np.zeros((self.dim,), dtype = np.int8)


    def create_bipolar_mem(self, numitem, dim):
        # Creates random bipolar memory of given size

        rand_arr = np.rint(np.random.rand(numitem, dim)).astype(np.int8)
        return (rand_arr*2 - 1)

    def hd_mul(self, A, B):
        # Return element-wise multiplication between bipolar HD vectors
        # inputs:
        #   - A: bipolar HD vector
        #   - B: bipolar HD vector
        # outputs:
        #   - A*B: bipolar HD vector
        return np.multiply(A,B,dtype = np.int8)

    def hd_threshold(self, A):
        # Given integer vector, threshold at zero to bipolar
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - [A]: bipolar HD vector
        return (np.greater_equal(A,0, dtype=np.int8)*2-1)

    def hd_protect(self, A):
        # Apply circular shift to vector
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - [A]: bipolar HD vector

        return np.roll(A,1,axis=0)

    def search_actuator_vals(self, A):
        # Find the nearest item in 'hd_actuator_vals' according to Hamming distance
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - i: integer index of closest item in 'hd_actuator_vals'
        dists = np.matmul(self.hd_actuator_vals, A, dtype = np.int)
        return np.argmax(dists)

    def encode_sensors(self, sensor_in):
        # Encode sensory data into HD space
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector

        sensor_vec = np.zeros((self.dim,), dtype = np.int8)
        for i,sensor_val in enumerate(sensor_in):
            binded_sensor = self.hd_mul(self.hd_sensor_ids[i,:],self.hd_sensor_vals[sensor_val,:])
            sensor_vec = sensor_vec + binded_sensor

        #return sensor_vec
        #return self.hd_threshold(sensor_vec)
        return self.hd_protect(self.hd_threshold(sensor_vec))

    def train_sample(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        sensor_vec = self.encode_sensors(sensor_in)
        act_vec = self.hd_actuator_vals[act_in,:]
        sample_vec = self.hd_mul(sensor_vec,act_vec)

        return sample_vec

    def test_sample(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        sensor_vec = self.encode_sensors(sensor_in)
        #print('Sensor vec:')
        #print(sensor_vec[1:10])
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec)
        #print('Unbind vec:')
        #print(unbind_vec[1:10])
        act_out = self.search_actuator_vals(unbind_vec)

        return act_out

    def train_from_file(self, file_in, threshold=False):
        # Build the program HV from a text file of recorded moves
        # inputs:
        #   -file_in: filename for the recorded moves
        # Currently, the hd_program_vec is not being thresholded
        game_data = np.loadtxt(file_in, dtype = np.int8, delimiter=',')
        sensor_vals = game_data[:,:-1]
        actuator_vals = game_data[:,-1]
        n_samples = game_data.shape[0]

        program_vec_b4thresh = np.zeros((self.dim,),dtype=np.int8)

        for sample in range(n_samples):
            sample_vec = self.train_sample(sensor_vals[sample,:],actuator_vals[sample])
            program_vec_b4thresh = program_vec_b4thresh + sample_vec

        if threshold:
            self.hd_program_vec = self.hd_threshold(program_vec_b4thresh)
        else:
            self.hd_program_vec = program_vec_b4thresh

        print('hd_program_vec:')
        print(self.hd_program_vec[1:10])

        return


    def train_live(self):
        return

    def test_live(self):
        return
