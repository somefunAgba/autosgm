# Tutorial adapted: https://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html

import math
import os
from typing import Optional, Tuple, TypeVar, List, Dict, Protocol
from collections import defaultdict, deque
from queue import Queue, PriorityQueue, LifoQueue

import numpy as np
from numpy import Infinity

# import colorama
from queue import PriorityQueue

q = PriorityQueue()

q.put((4, 'Read'))
q.put((2, 'Play'))
q.put((5, 'Write'))
q.put((1, 'Code'))
q.put((3, 'Study'))

while not q.empty():
    next_item = q.get()
    print(next_item)

# Graph and Location

Location = TypeVar('Location')
class Graph(Protocol):
    def neighbors(self, id:Location) -> list[Location]: 
        pass
    
    def cost(self, from_node, to_node):
        pass
    
    def heuristic(self, node, goal, type):
        pass
        
class aGraph:
    def __init__(self) -> None:
        self.edges: dict[Location, list[Location]] = {}
        self.weights: dict[(Location,Location), float] = {}
        
    def neighbors(self, id:Location) -> list[Location]:
        return self.edges[id]
    
    def cost(self, from_node, to_node):
        return self.weights[(from_node, to_node)]


    
mygraph = aGraph()
mygraph.edges = {
    'A': ['B'],
    'B': ['C'],
    'C': ['B', 'D','F'],
    'D': ['C', 'E'],
    'E': ['F'],
    'F': [],
}    

for current in mygraph.edges:
    for next in mygraph.edges[current]:
        mygraph.weights[(current,next)] = 1
    

def bfs(graph:Graph, start:Location):
    frontier = Queue()
    frontier.put(start)
    reached = defaultdict(bool)
    reached[start] = True 
    
    while not frontier.empty():
        current:Location = frontier.get()
        print("visiting %s"%(current))
        for next in graph.neighbors(current):
            if next not in reached:
                frontier.put(next)
                reached[next] =True 
                
    return reached
                
bfs(mygraph, 'D')


# Grid

GridLocation = Tuple[int, int]

class SquareGrid:
    def __init__(self, width:int=3, height:int=3):
        self.width = width
        self.height = height
        self.walls: List[GridLocation] = []
        self.weights: dict[(Location,Location), float] = {}

        
    def in_bnds(self, id:GridLocation):
        (x,y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id:GridLocation):
        return id not in self.walls
    
    def neighbors(self, id:GridLocation):
        (x,y) = id
        neighbors = [(x+1,y), (x-1,y), (x,y-1), (x,y+1)] # to: E W N S
        # ugly path fix
        if (x+y) % 2 == 0: neighbors.reverse() # S N W E
        # filter invalid neighbors
        neighbors = filter(self.in_bnds, neighbors)
        neighbors = filter(self.passable, neighbors)
        
        # returns an Iterator[GridLocation]
        return neighbors
    
    
    def cost(self, from_node:GridLocation, to_node:GridLocation):
        self.weights[(from_node,to_node)] = 1
        
        cost = self.weights[(from_node,to_node)]
        
        # slight diagonal movement hack
        (x1,y1) = from_node
        (x2,y2) = to_node
        eps = 0
        if (x1+y1) % 2 == 0 and x1!=x2: eps = 1e-3
        if (x1+y1) % 2 == 1 and y1!=y2: eps = 1e-3 # 1e-6

        return cost + eps
    
    def heuristic(self, node:GridLocation, goal:GridLocation, type='man'):
        (x1,y1), (x2,y2) = node, goal
        
        h = 0
        if type == 'man':
            h = abs(x1-x2) + abs(y1-y2)
            
        if type == 'eul':
            h = (0.5*( ((x1-x2)**2) + ((y1-y2)**2) ))
            h = math.sqrt(h)
            
        # movement hack: move with mid-point, 
        # (works  bettter and without the other cost hack)
        cx = 0.5*(x1+x2)
        cy = 0.5*(y1+y2)
        he = 0.5*( abs(y1-cy) + abs(x1-cx) )
            
        h = h + (1e-2*he)  #+ 1e-3*abs(np.random.randn())
        
        return h
        

termcol= {'red':'\033[91m', 'green':'\033[92m',
        'yellow':'\x1b[93m', 'blue':'\033[94m',
        'purple':'\033[95m', 'cyan':'\033[96m',
        'gray':'\033[90m', 'end':'\033[0m\033[39m',
        }  # change it, according to the color need

emjis = {'sqsq': '▣', 'sqdt': '⊡', 'sqx':'⊠', 'trx':'⨻',
          'cdt':'⨀', 'cc':'⊚', 'ch':'⊝', 'cnull':'⊘', 'cstr':'⊛',
          'xbig':'⨉', 'xsml':'⨯', 'str':'⋆', 'dt':'⋅',
          'edir':'→', 'wdir':'←', 'ndir':'↑', 'sdir':'↓',
          1:'①', 4:'④', 7:'⑦', 2:'②', 3:'③', 
          6:'⑥', 9:'⑨', 8:'⑧', 5:'⑤', 0:'○', 
          'cpls':'⊕', 'pls':'+', 'vl':'|', 'hl':'—', 'dtc':'•', 'dtr':'▪',
          }

def draw_grid(g:SquareGrid, start=None, goal=None, spath=None, start2goal_path=None, costs=None):
    
    def get_cellstr(rw,cl):
        # transform (row, col) id to grid coordinates (x,y) id
        x, y = (cl,(g.height-1)-rw)
        cidx = (x,y)
        # empty
        cellstr = emjis['dt']
        swdt = 0
        # types, and from: right, left, up, down
        cfrom = [(x+1,y), (x-1,y), (x,y-1), (x,y+1)] # grid: W E N S
        if spath is not None:
            dval = ""
            if costs is not None and cidx in costs:
                dval = r" %.2g"%(costs[cidx])
                swdt += len(dval)
                dval = termcol['gray']+dval
            
            if start is not None and cidx == start:
                cellstr = termcol['cyan']+emjis['cdt']+dval+termcol['end']
                swdt += 1
            elif goal is not None and cidx == goal:
                cellstr = termcol['green']+emjis['sqdt']+dval+termcol['end']
                swdt += 1
            elif len(g.walls) and cidx in g.walls:
                cellstr = termcol['red']+emjis['sqx']+termcol['end']
                swdt += 1
            else: 
                if cidx in spath:
                    tcol =  termcol['gray']
                    if start2goal_path is not None and cidx in start2goal_path:
                        tcol = termcol['green']
                    
                    if spath[cidx] == cfrom[0]:
                        cellstr = tcol + emjis['wdir']+dval+termcol['end']
                    elif spath[cidx] == cfrom[1]:
                        cellstr = tcol+emjis['edir']+dval+termcol['end']
                    elif spath[cidx] == cfrom[2]:
                        cellstr = tcol+emjis['ndir']+dval+termcol['end']
                    elif spath[cidx] == cfrom[3]:
                        cellstr = tcol+emjis['sdir']+dval+termcol['end']
        else:
            swdt += 1
            
        return cellstr, swdt
        
    dims = (g.height,g.width) # rows (height), cols (width)
    celsp = 9
    hdft = termcol['blue']+emjis['pls']
    for _ in range(dims[1]):
        hdft += f"{emjis['hl']*celsp}"+ emjis['pls']
    hdft += termcol['end']
        
    print(hdft)    
    stremp = " "
    for rw in range(dims[0]):
        onerow = emjis['vl']
        for cl in range(dims[1]):
            cstr, swdt = get_cellstr(rw,cl)
            rmsp = (celsp-swdt)//2
            onerow += f"{stremp*rmsp}{cstr}{stremp*rmsp}"+ emjis['vl']
        print(onerow)
        print(hdft)
        
def bfs2(graph:Graph, start:Location, goal:Location=None):
    frontier = Queue()
    frontier.put(start)
    reached = defaultdict(Optional[Location])
    reached[start] = None 
    
    while not frontier.empty():
        current:Location = frontier.get()
        if current == goal: break
        # print("visiting %s"%(current))
        for next in graph.neighbors(current):
            if next not in reached:
                frontier.put(next)
                reached[next] = current

    # reconstruct path
    if goal is not None and goal in reached:
        current = goal
        io_path = deque([current,])
        while current is not None:
            next = reached[current]
            io_path.appendleft(next)
            current = next
    else:
        io_path = None
                        
    return reached, io_path

def dfs(graph:Graph, start:Location, goal:Location=None):
    frontier = LifoQueue()
    frontier.put(start)
    reached = defaultdict(Optional[Location])
    reached[start] = None 
    
    while not frontier.empty():
        current = frontier.get()
        if current == goal: break
        # print("visiting %s"%(current))
        for next in graph.neighbors(current):
            if next not in reached:
                frontier.put(next)
                reached[next] = current

    # reconstruct path
    if goal is not None and goal in reached:
        current = goal
        io_path = deque([current,])
        while current is not None:
            next = reached[current]
            io_path.appendleft(next)
            current = next
    else:
        io_path = None
                        
    return reached, io_path

def dijkstra_ucs(graph:Graph, start:Location, goal:Location=None):
    frontier = PriorityQueue()
    frontier.put((0,start))
    reached = defaultdict(Optional[Location])
    reached[start] = None 
    cost_sofar = defaultdict(Optional[Location])
    cost_sofar[start] = 0 
    
    while not frontier.empty():
        item = frontier.get()
        current = item[1]
        if current == goal: break
        
        for next in graph.neighbors(current):
            new_cost = cost_sofar[current] + graph.cost(current,next)
            # if next not in cost_sofar or new_cost < cost_sofar[next]:
            if new_cost < cost_sofar.get(next, Infinity):
                cost_sofar[next] = new_cost
                priority = new_cost
                frontier.put((priority,next))
                reached[next] = current
      
    # reconstruct path
    if goal is not None and goal in reached:
        current = goal
        io_path = deque([current,])
        while current is not None:
            next = reached[current]
            io_path.appendleft(next)
            current = next
    else:
        io_path = None
              
    return reached, cost_sofar, io_path

def a_star(graph:Graph, start:Location, goal:Location=None):
    frontier = PriorityQueue()
    # the hueristic additon can be skipped here
    priority = 0 + graph.heuristic(start, goal) 
    frontier.put((priority,start))
    reached = defaultdict(Optional[Location])
    reached[start] = None 
    cost_sofar = defaultdict(Optional[Location])
    cost_sofar[start] = 0 
    
    while not frontier.empty():
        item = frontier.get()
        current = item[1]
        if current == goal: break
        
        for next in graph.neighbors(current):
            new_cost = cost_sofar[current] + graph.cost(current,next)
            # if next not in cost_sofar or new_cost < cost_sofar[next]:
            if new_cost < cost_sofar.get(next, Infinity):
                cost_sofar[next] = new_cost
                priority = new_cost + graph.heuristic(next, goal)
                frontier.put((priority,next))
                reached[next] = current
      
    # reconstruct path
    if goal is not None and goal in reached:
        current = goal
        io_path = deque([current,])
        while current is not None:
            next = reached[current]
            io_path.appendleft(next)
            current = next
    else:
        io_path = None
            
    return reached, cost_sofar, io_path

print('GRID')
g = SquareGrid(3, 3)
draw_grid(g)

print('BFS: START')
g = SquareGrid(3, 3)
# g.walls += [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4)]
start = (1,2)
spath, _ = bfs2(g, start)
draw_grid(g, start=start, spath=spath)

print('DFS: START')
g = SquareGrid(3, 3)
# g.walls += [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4)]
start = (1,2)
spath, _ = dfs(g, start)
draw_grid(g, start=start, spath=spath)

print('DFS: START-GOAL')
g = SquareGrid(8, 6)
g.walls += [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (4,2), (4,3), (4,4),]
start = (1,2)
goal = (6,0)
spath, s2gpath = dfs(g, start, goal)
draw_grid(g, start=start, goal=goal, spath=spath, start2goal_path=s2gpath)

print('BFS: START-GOAL')
g = SquareGrid(8, 6)
g.walls += [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (4,2), (4,3), (4,4),]
start = (1,2)
goal = (6,0)
spath, s2gpath = bfs2(g, start, goal)
draw_grid(g, start=start, goal=goal, spath=spath, start2goal_path=s2gpath)

print('UCS: START-GOAL')
g = SquareGrid(8, 6)
g.walls += [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (4,2), (4,3), (4,4),]
start = (1,2)
goal = (6,0)
spath, cost_sofar, s2gpath = dijkstra_ucs(g, start, goal)
draw_grid(g, start=start, goal=goal, spath=spath, start2goal_path=s2gpath, costs=cost_sofar)


print('A-STAR: START-GOAL')
g = SquareGrid(8, 6)
g.walls += [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (4,2), (4,3), (4,4),]
start = (1,2)
goal = (6,0)
spath, cost_sofar, s2gpath = a_star(g, start, goal)
draw_grid(g, start=start, goal=goal, spath=spath, start2goal_path=s2gpath, costs=cost_sofar)
