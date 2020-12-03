from modules.game_module import game_module

train_file = './data/spanning_dat.out'

live = int(input("Enter 1 to play the game live, 0 to autoplay using recall\n"))
gametype = int(input("Enter 1 to play with goals, 0 without\n"))

a = game_module()
a.setup_game()
a.train_from_file(train_file)
if live:
    a.play_game(gametype)
else:
    a.autoplay_game(gametype)

