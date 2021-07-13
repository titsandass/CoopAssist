import unittest

class Node:
    def __init__(self,x,y,cmp_x):
        '''
        x,y  -  point
        cmp_x - compare point by x-coord if True else compare by y-coord
        '''
        self.x = x
        self.y = y
        self.cmp_x = cmp_x
        self.left = None
        self.right = None

    def __lt__(self,other):
        if self.cmp_x:
            if self.x != other.x:
                return self.x < other.x
            else:
                return self.y < other.y
        else:
            if self.y != other.y:
                return self.y < other.y
            else:
                return self.x < other.x

    def __eq__(self,other):
        if other.x == self.x and other.y == self.y:
            return True
        else:
            return False

class Tdtree:
    def __init__(self):
        self.root = None

    def search(self,key):
        path = list()
        self.curnode = self.root
        while self.curnode is not None:
            if not self.curnode == key:
                if self.curnode < key:
                    self.curnode = self.curnode.right
                else:
                    self.curnode = self.curnode.left
            else:
                return True
        return False

    def insert(self,key):
        self.curnode = self.root
        prevnode = None
        if self.curnode is not None:
          while self.curnode is not None:
              if not self.curnode == key:
                  prevnode = self.curnode
                  if self.curnode < key:
                      self.curnode = self.curnode.right
                  else:
                      self.curnode = self.curnode.left
              else:
                  return False

          if prevnode.left == self.curnode:
              key.cmp_x = not prevnode.cmp_x
              prevnode.left = key
        else:
            self.root = key
            key.cmp_x = True

class TestNodes(unittest.TestCase):

    def setUp(self):
        self.n1 = Node(3,5,True)
        self.n2 = Node(6,1,False)

    def test_comparison(self):
        '''
        '''

        self.assertIs(self.n1 < self.n2,True)
        self.assertIs(self.n2 < self.n1,True)
        self.assertNotEqual(self.n1, self.n2)


class TestTdTree(unittest.TestCase):

    def setUp(self):
        self.t = Tdtree()
        self.t.root = Node(5,2,True)
        self.t.root.left = Node(1,4,False)
        self.t.root.right = Node(6,3,False)
        self.n1 = Node(6, 3, False)
        self.n2 = Node(5, 3, True)

    def testSearch(self):

        self.assertIs(self.t.search(self.n1),True)
        self.assertIs(self.t.search(self.n2),False)

    def testInsert1(self):
        self.t.insert(self.n2)
        self.assertEqual(self.t.search(self.n2),True)
        self.assertEqual(self.t.root.right.left,self.n2)

if __name__ == '__main__':
    unittest.main(verbosity=2)