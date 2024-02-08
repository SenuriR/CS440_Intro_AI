import random
grid = [[0 for i in range(10)] for j in range(10)]
grid = [[random.randint(0,1) for i in range(10)] for j in range(10)]
grid[0][0] = 0
grid[9][9] = 0

for i in range(10):
    for j in range(10):
        print(grid[i][j], end = ' ')
    print()