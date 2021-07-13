class Node:
    def __init__(self, start_time, end_time, sat_pair):
        self._start_time    = start_time
        self._end_time      = end_time

        self._sat_pair      = sat_pair

        self._left          = None
        self._right         = None
        self._depth         = 1

    def __lt__(self, node):
        if self._depth%2 == 1:
            return self._start_time <= node._start_time
        else:
            return self._end_time <= node._end_time

class SATPairTree:
    def __init__(self, root_node):
        self._root_node = root_node

    def insert(self, inserting_node):
        curr_node = self._root_node
        left = None
        right = None
        while curr_node is not None:
            if inserting_node < curr_node:
                left = curr_node
                right = None

                curr_node = curr_node._left
                inserting_node._depth += 1
            else:
                left = None
                right = curr_node
                
                curr_node = curr_node._right
                inserting_node._depth += 1
        if left is not None:
            left._left = inserting_node
        elif right is not None:
            right._right = inserting_node
    
    def search(self, searching_time):
        path = list()
        curr_node = self._root_node
        while curr_node is not None:
            if curr_node._depth%2 == 1:
                ###x값 비교###
                if searching_time < curr_node._start_time:
                    curr_node = curr_node._left
                else:
                    if searching_time <= curr_node._end_time:
                        path.append(curr_node._sat_pair)
                    curr_node = curr_node._right
            else:
                ###y값 비교###
                if searching_time <= curr_node._end_time:
                    if searching_time >= curr_node._start_time:
                        path.append(curr_node._sat_pair)
                    curr_node = curr_node._left
                else:
                    curr_node = curr_node._right
        return path

if __name__ == '__main__':
    a = Node(0, 20, [1,     2])
    b = Node(0, 40, [3,	4])
    c = Node(0, 11760, [5,    6])
    d = Node(0, 40, [7,	8])
    e = Node(0, 100, [9,	10])
    f = Node(0, 10, [11,	12])

    tree = SATPairTree(a)
    tree.insert(b)
    tree.insert(c)
    tree.insert(d)
    tree.insert(e)
    tree.insert(f)
    path = tree.search(10)

    pass