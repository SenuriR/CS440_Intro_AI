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
            if grid[i-1][j]:
                if grid[i-1][j] == 0:
                    node1 = Node((i-1, j))
                    node.next_node = node1
                    node1.parent = (i,j)
            if grid[i][j-1]:
                if grid[i][j-1] == 0:
                    node2 = Node((i, j-1))
                    node.next_node = node1
                    node1.parent = (i,j)
            if grid[i+1][j]:
                if grid[i+1][j] == 0:
                    node3 = Node((i+1, j))
                    node.next_node = node1
                    node1.parent = (i,j)
            if grid[i][j+1]:
                if grid[i][j+1] == 0:
                    x = 0

neighbors = {}
neighbors.add(node1)
neighbors.add(node2)
neighbors.add(node3)
neighbors.add(node4)

for neighbor in neighbors:
    if neighbor:
        # this neighbor state is not blocked
        x = 0
    else:
        # this neighbor state is blocked
        x = 0

# separate calculations of h values for cells on grid here...
def manhattan(initialCell, goalCell):
    origin = initialCell.data
    goal = goalCell.data
    return abs(origin[0] - goal[0]) + abs(origin[1] - goal[1])