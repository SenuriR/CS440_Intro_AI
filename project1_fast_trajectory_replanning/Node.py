class Node:
    def __init__(self, x=None, y=None, next_node=None,
    i=None, j=None, g_value=0, h_value=0):
        self.data = [x, y]
        self.next_node = next_node if next_node is not None else []
        self.parent = None
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = self.g_value + self.h_value

    def __lt__(self, other):
        return self.f_value < other.f_value