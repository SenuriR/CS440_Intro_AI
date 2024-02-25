'''
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
'''

