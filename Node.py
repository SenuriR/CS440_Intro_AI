class Node:
    def __init__(self, x=None, y=None, next_node=None,
    i=None, j=None, g_value=None, h_value=None, f_value=None):
        self.data = [x, y]
        self.next_node = []
        self.parent = [i, j]
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = f_value