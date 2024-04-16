import sys
from heapq import heappush,heappop,heapify

class BinHeap:
    
    def __init__(self, maxsize): 
        self.maxsize = maxsize 
        self.size = 0
        self.Heap = [0]*(self.maxsize + 1) 
        self.Heap[0] = -1 * sys.maxsize 
        self.FRONT = 1
        
    # returns floor division of current node position to get parent nodes position
    def parent( selfNode, position):
        return (position-1)/2
    
    # insert key with value key
    def insertKey(selfNode, k):
        heappush(self.heap,k)
    
    def decreaseKey(self,i,new_val):
        self.heap[i] = new_val
        while(i != 0 and self.heap[self.parent(i)] > self.heap[i]):
            self.heap[i] , self.heap[self.parent(i)] = (self.heap[self.parent(i), self.heap[i]])

    def extractMin(self):
        return heappop(self.heap)
    
    def deleteKey(self,i):
        self.decreaseKey(i,float("-inf"))
        self.extractMin()
        
    def getMin(self):
        return self.heap[0]    