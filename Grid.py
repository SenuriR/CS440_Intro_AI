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
                    node.next_node.append(node1)
                    node1.parent[0], node1.parent[1] = i,j
            if grid[i][j-1]:
                if grid[i][j-1] == 0:
                    node2 = Node((i, j-1))
                    node.next_node.append(node2)
                    node2.parent[0], node2.parent[1] = i,j
            if grid[i+1][j]:
                if grid[i+1][j] == 0:
                    node3 = Node((i+1, j))
                    node.next_node.append(node3)
                    node3.parent[0], node3.parent[1] = i,j
            if grid[i][j+1]:
                if grid[i][j+1] == 0:
                    node4 = Node((i, j+1))
                    node.next_node.append(node4)
                    node4.parent[0], node4.parent[1] = i,j

        compare = []
        for node in node.next_node:
            node.g_value = g
            node.h_value = manhattan(node)
            node.f_value = node.g_value + node.h_value
            compare.append(node)
        
        sorted_nodes = sorted(compare, key=lambda node: node.f_value, reverse=True)
        heap.add(sorted_nodes[0]) # edit once heap is finalized
        heap.add(sorted_nodes[1])
        
        g += 0

# separate calculations of h values for cells on grid here...
def manhattan():
    # something
    x = 0