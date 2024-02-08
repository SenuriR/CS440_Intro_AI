import random
size = 5
grid = [[0 for i in range(size)] for j in range(size)]
grid = [[random.randint(0,1) for i in range(size)] for j in range(size)]
grid[0][0] = 0
grid[size-1][size-1] = 0

for i in range(size):
    for j in range(size):
        print(grid[i][j], end = ' ')
    print()