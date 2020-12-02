from modules.hd_module import hd_module

train_file = './data/spanning_dat.out'
test_file = './data/spanning_dat.out'

a = hd_module()
a.train_from_file(train_file)
print(a.hd_program_vec)
a.test_from_file(test_file)
