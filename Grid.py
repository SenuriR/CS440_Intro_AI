import random
from Node import Node
import binHeap
import heapq

updatedHeuristicsMap = {}
# I think we may need to have a better way of tracking g values

def updatedHeuristics(finalNode):
  curr = finalNode
  while curr is not None:
    updatedHeuristicsMap[(curr.data[0],curr.data[1])] = finalNode.g_value - curr.g_value
    curr = curr.parent


def getUpdatedHeuristic(node,targetNode):
  if (node.data[0],node.data[1]) in updatedHeuristicsMap:
    return updatedHeuristicsMap[(node.data[0],node.data[1])]
  else:
    return manhattan(node,targetNode)


def manhattan(initialCell, goalCell):
    return abs(initialCell.data[0] - goalCell.data[0]) + abs(initialCell.data[1] - goalCell.data[1]) # goalCell is used twice since x and y will always be the same (size 5)

# I wrote the repeated forwards and backwards methods below based off of this method
# so I think we can comment out aStar()
def aStar(grid, start, target): #grid must be 2d array, start and target must be 1d arrays of two elements

    startNode = Node(start[0],start[1],None,None,None,0,manhattan(Node(start[0],start[1]),Node(target[0],target[1])))
    targetNode = Node(target[0],target[1])

    openList = [] #open list
    heapq.heappush(openList, (startNode.f_value,startNode))
    closed = set() #closed list
    while openList:
      currNode = heapq.heappop(openList)

      if tuple(currNode.data) in closed: # Cannot append arrays to an sets, so i made it into a tuple
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

def repeatedForwardsAStar(grid, start, target):
  # initialize start and target nodes
  startNode = Node(start[0],start[1],None,None,None,0,manhattan(Node(start[0],start[1]),Node(target[0],target[1])))
  targetNode = Node(target[0],target[1])
  openList = [] #open list
  heapq.heappush(openList,(startNode.f_value, startNode))
  closed = set() #closed list -- here it contains nodes
  while openList:
    currNode = heapq.heappop(openList)
    if currNode == targetNode: # Path found
        finalPath = []
        while currNode: # While a parent exists for the node
          finalPath.append(currNode.data)
          currNode = currNode.parent
        return finalPath[::1] # Returning reversed path so that the array starts at startingNode and ends with targetNode
    # closed.add(tuple(currNode.data))
    closed.add(currNode)
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]: # All neighbors for current cell
        x = currNode.data[0] + dx
        y = currNode.data[1] + dy
        neighborNode = Node(x, y)
        tmp_g = currNode.g_value + manhattan(currNode, neighborNode)

        # if the x,y pair is out of bounds or the cell is blocked
        if(0 <= x < grid[0].length()) and (0 <= y < grid[1].length()):
          if grid[x][y] == 1:
            continue
        else:
          continue

        # tie breaker
        matching_node = next(node for node in closed if node.x == neighborNode.x and node.y == neighborNode.y)
        # ^ I had to change the above line bc the "in" comparator was comparing references, not the specific attributes we care about
        # ^ this change might effect efficiency
        if matching_node.g_value >= tmp_g: # the existing node in closed won
          continue;
        if matching_node.g_value > tmp_g: # the new neighbor version won
          neighborNode.parent = currNode
          neighborNode.g_value = tmp_g # cost we've been through
          neighborNode.f_value = tmp_g + manhattan(neighborNode, targetNode)
          heapq.heappush(openList, (neighborNode.f_value, neighborNode))
    return False
    
def repeatedBackwardsAStar(grid, start, target):
  # initialize start and target nodes
  startNode = Node(start[0],start[1],None,None,None,0,manhattan(Node(start[0],start[1]),Node(target[0],target[1])))
  targetNode = Node(target[0],target[1])
  openList = [] #open list
  heapq.heappush(openList,(targetNode.f_value, targetNode))
  closed = set() #closed list -- here it contains nodes
  while openList:
    currNode = heapq.heappop(openList)
    if currNode.data == targetNode.data: # Path found
        finalPath = []
        while currNode: # While a parent exists for the node
          finalPath.append(currNode.data)
          currNode = currNode.parent
        return finalPath[::1] # Returning reversed path so that the array starts at startingNode and ends with targetNode
    # closed.add(tuple(currNode.data))
    closed.add(currNode)
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]: # All neighbors for current cell
        x = currNode.data[0] + dx
        y = currNode.data[1] + dy
        neighborNode = Node(x, y)
        tmp_g = currNode.g_value + manhattan(currNode, neighborNode)

        # if the x,y pair is out of bounds or the cell is blocked
        if(0 <= x < grid[0].length()) and (0 <= y < grid[1].length()):
          if grid[x][y] == 1:
            continue
        else:
          continue

        # tie breaker
        matching_node = next(node for node in closed if node.x == neighborNode.x and node.y == neighborNode.y)
        # ^ I had to change the above line bc the "in" comparator was comparing references, not the specific attributes we care about
        # ^ this change might effect efficiency
        if matching_node.g_value >= tmp_g: # the existing node in closed won
          continue;
        if matching_node.g_value > tmp_g: # the new neighbor version won
          neighborNode.parent = currNode
          neighborNode.g_value = tmp_g # cost we've been through
          neighborNode.f_value = tmp_g + manhattan(neighborNode, targetNode)
          heapq.heappush(openList, (neighborNode.f_value, neighborNode))
    return False

def adaptiveA(grid, start, target): # made a few edits here
    startNode = Node(start[0],start[1],None,None,None,0,manhattan(Node(start[0],start[1]),Node(target[0],target[1])))
    targetNode = Node(target[0],target[1])

    openList = [] #open list
    heapq.heappush(openList,(startNode.f_value, startNode))
    closed = set() #closed list -- here it contains the (x,y) tuple...
    while openList:
      currNode = heapq.heappop(openList)

      if tuple(currNode.data) in closed: # Cannot append arrays to an sets, so i made it into a tuple
        continue

      if currNode.data == targetNode.data: # Path found
        updatedHeuristics(currNode) # Finds each parent for the current node (which is target) and stores their actual distance from node to target in map for possible later use
        finalPath = []
        while currNode: # While a parent exists for the node
          finalPath.append(currNode.data)
          currNode = currNode.parent
        return finalPath[::1] # Returning reversed path so that the array starts at startingNode and ends with targetNode

      closed.add(tuple(currNode.data))

      for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]: # All neighbors for current cell
        x = currNode.data[0] + dx
        y = currNode.data[1] + dy
        neighborNode = Node(x, y)
        tmp_g = currNode.g_value + manhattan(currNode, neighborNode)

        # if the x,y pair is out of bounds or the cell is blocked
        if(0 <= x < grid[0].length()) and (0 <= y < grid[1].length()):
          if grid[x][y] == 1:
            continue
        else:
          continue

        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0:
          neighbor_h_value = getUpdatedHeuristic(Node(x,y),targetNode) # Attempts to see if the previous distance has been calculated already, if not: manhattan distance, if so returns the accurate heuristic
          neighborNode.h_value = neighbor_h_value
          neighborNode.parent = currNode
          neighborNode.g_value = tmp_g # cost we've been through
          neighborNode.f_value = tmp_g + manhattan(neighborNode, targetNode)
          heapq.heappush(openList, (neighborNode.f_value, neighborNode))

    return None #if no path is found

size = 101
grid = [[0 for i in range(size)] for j in range(size)] #All cells unvisited
grid = [[random.choices([0, 1], weights=[0.7, 0.3], k=1)[0] for i in range(size)] for j in range(size)] #Marking random cells as visited and unblocked
grid[0][0] = 0
grid[size-1][size-1] = 0

for i in range(size):
    for j in range(size):
        print(grid[i][j], end = ' ')
    print()
