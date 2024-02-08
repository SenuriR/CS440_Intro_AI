import random
from Node import Node
size = 5
grid = [[0 for i in range(size)] for j in range(size)]
grid = [[random.randint(0,1) for i in range(size)] for j in range(size)]
grid[0][0] = 0
grid[size-1][size-1] = 0

for i in range(size):
    for j in range(size):
        print(grid[i][j], end = ' ')
    print()

nodes = {}
g = 0 # g is traveled distance/cost

for i in range(size):
    for j in range(size):
        if grid[i][j] == 0:
            node = Node((i, j))
            if grid[i-1][j] == 0:
                node1 = Node((i-1, j))
                node.nextNode.add(node1)
                node1.parent = (i,j)
            if grid[i][j-1] == 0:
                x
            if grid[i+1][j] == 0:
                x
            if grid[i][j+1] == 0:
                x

# separate calculations of h values for cells on grid here...
def manhattan():
    x