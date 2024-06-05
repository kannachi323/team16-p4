"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
   
    # *** Your Code Here ***
    """
    marked = False * graph size
    def dfs(graph, v):
        stack = [v]
        while len(stack) > 0
            v = stack.pop()
            if not marked[v]:
                visit(v)
                marked[v] = True
                for w in G.neighbors(v):
                    if not marked[w]:
                        stack.append(w)
    """

    path = []
    visited = set()
    stack = Stack()
    stack.push((problem.startingState(), []))
    
    while not stack.isEmpty():
        state, path = stack.pop()

        if problem.isGoal(state):
            return path
        
        if state not in visited:
            visited.add(state)
            for successor in problem.successorStates(state):
                next_state, direction = successor[0], successor[1]
                if next_state not in visited:
                    stack.push((next_state, path + [direction]))
    
    return None
    
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
   
    # *** Your Code Here ***
    path = []
    visited = []
    queue = Queue()
    queue.push((problem.startingState(), []))
    
    while not queue.isEmpty():
        state, path = queue.pop()
        print(state)

        if problem.isGoal(state):
            return path

        if state not in visited:
            visited.append(state)
            for successor in problem.successorStates(state):
                next_state, direction = successor[0], successor[1]
                print(direction)
                if next_state not in visited:
                    queue.push((next_state, path + [direction]))
                    
    return None

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    path = []
    visited = set()
    pq = PriorityQueue()
    pq.push((problem.startingState(), [], 0), 0)
    
    while not pq.isEmpty():
        state, path, curr_cost = pq.pop()

        if problem.isGoal(state):
            return path
        
        if state not in visited:
            visited.add(state)
            for successor in problem.successorStates(state):
                next_state, direction, next_cost = successor[0], successor[1], successor[2]
                if next_state not in visited:
                    total_cost = curr_cost + next_cost
                    pq.push((next_state, path + [direction], total_cost), total_cost)
    return None

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    path = []
    visited = []
    pq = PriorityQueue()
    pq.push((problem.startingState(), [], 0), 0)
    
    while not pq.isEmpty():
        state, path, curr_cost = pq.pop()

        if problem.isGoal(state):
            return path
        
        if state not in visited:
            visited.append(state)
            for successor in problem.successorStates(state):
                next_state, direction, next_cost = successor[0], successor[1], successor[2]
                if next_state not in visited:
                    path_cost = curr_cost + next_cost
                    total_cost = path_cost + heuristic(next_state, problem)
                    pq.push((next_state, path + [direction], total_cost), total_cost)
    
    return None