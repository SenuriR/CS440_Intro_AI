import random
from Node import Node

# separate calculations of h values for cells on grid here...
def manhattan(initialCell, goalCell):
    origin = initialCell.data
    return abs(origin[0] - goalCell) + abs(origin[1] - goalCell) #goalCell is used twice since x and y will always be the same (size 5)

def aStar(graph, startNode, targetNode):
    open = []
    open.push(startNode)
    closed = set() #set with all indices that have been searched for
    while !open.isEmpty():
        minCostNode = open.pop() #Lowest cost node / first node in priority q
        closed.add(minCostNode.data)

        if minCostNode == targetNode: #If target node is reached
            finalPath = []
            currNode = minCostNode
            while currNode is not None:
                finalPath.append(currNode.data)
                currNode = currNode.parent
            return finalPath[::-1] #Reverses array to get from start to finish
        
        for child in minCostNode.next_node: #Assuming all children are unblocked and arrays of indices
            if child.data in closed: #If child has been visited 
                continue
            
            #Checking to see if new path to children is shorter
            shouldSkip = False 
            for node in open:
                if node.data == child.data:
                    if node.g_value < child.g_value: 
                        shouldSkip = True
                        break
            if shouldSkip:
                continue

            open.push(child) #if potential shorter path is found

    return None #if no path is found

size = 5
grid = [[0 for i in range(size)] for j in range(size)] #All cells unvisited
grid = [[random.randint(0,1) for i in range(size)] for j in range(size)] #Marking random cells as visited and unblocked
grid[0][0] = 0
grid[size-1][size-1] = 0

for i in range(size):
    for j in range(size):
        print(grid[i][j], end = ' ')
    print()

nodes = {}
g = 1 # g is traveled distance/cost

for i in range(size): #Searching through every neighbor and creating a graph with nodes, and assigning values to each node
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
            node.h_value = manhattan(node,size-1)
            node.f_value = node.g_value + node.h_value
            compare.append(node)
        
        sorted_nodes = sorted(compare, key=lambda node: node.f_value, reverse=True)
        heap.add(sorted_nodes[0]) # edit once heap is finalized
        heap.add(sorted_nodes[1])
        
        g += 0


