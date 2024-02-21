import random
from Node import Node
import binHeap
import heapq

def manhattan(initialCell, goalCell):
    return abs(initialCell.data[0] - goalCell.data[0]) + abs(initialCell.data[1] - goalCell.data[1]) # goalCell is used twice since x and y will always be the same (size 5)

def aStar(grid, start, target): #grid must be 2d array, start and target must be 1d arrays of two elements
    
    startNode = Node(start[0],start[1],None,None,None,0,manhattan(Node(start[0],start[1]),Node(target[0],target[1])))
    targetNode = Node(target[0],target[1])

    openList = [] #open list
    heapq.heappush(openList,startNode)
    closed = set() #closed list
    while openList:
      currNode = heapq.heappop(openList)

      if tuple(currNode.data) in closed:
        continue

      if currNode.data == targetNode.data: # Path found
        finalPath = []
        while currNode: # While a parent exists for the node
          finalPath.append(currNode.data)
          currNode = currNode.parent
        return finalPath[::1] # Returning reversed path so that the array starts at startingNode and ends with targetNode

      closed.add(tuple(currNode.data))

      for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]: # All neighbors for current cell
        x = currNode.data[0] + dx
        y = currNode.data[1] + dy

        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0:
          neighbor = Node(x,y,None,None,None,currNode.g_value + 1,manhattan(Node(x,y),targetNode))
          neighbor.parent = currNode

          if (x,y) not in closed:
            heapq.heappush(openList, neighbor)

    return None #if no path is found

size = 101
grid = [[0 for i in range(size)] for j in range(size)] #All cells unvisited
grid = [[random.randint(0,1) for i in range(size)] for j in range(size)] #Marking random cells as visited and unblocked
grid[0][0] = 0
grid[size-1][size-1] = 0

for i in range(size):
    for j in range(size):
        print(grid[i][j], end = ' ')
    print()

nodes = {}
heap = []
g = 0 # g is traveled distance/cost

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
        binHeap.insertKey(heap,sorted_nodes[0]) # edit once heap is finalized
        binHeap.insertKey(heap,sorted_nodes[1])
        
        g += 1


