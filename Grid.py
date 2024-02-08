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
def manhattan(int x1, int x2, int y1, int y2):
    return abs(x2 - x1) + abs(y2 - y1)
    
def f(int g, int h):
    h = manhattan(x1,x2,y1,y2)
    f = g + h