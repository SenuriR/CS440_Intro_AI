class Node:
    def __init__(self, x=None, y=None, next_node=None,
    parent=None, g_value=None, h_value=None, f_value=None):
        self.data = (x,y)
        self.next_node = next_node
        self.parent = parent
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = f_value

Node1 = Node(data=42)
print(Node1.data)