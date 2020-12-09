from modules.game_module import game_module

train_file = './data/game_dat.out'

a = game_module()
a.train_from_file(train_file)
a.test_game(1000)
