# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    def solve(stack: list, start):
        stack.append(start)
        start_pos = start[0]

        if problem.isGoalState(start_pos):  
            return stack
        
        sucessors = problem.getSuccessors(start_pos)
        
        for s in sucessors:
            pos, direction, _ = s
            if pos in [item[0] for item in stack]:
                continue
            new_stack = solve(stack.copy(), (pos, direction))
            if new_stack: return new_stack
        return None


    cur_pos = problem.getStartState()
    result = [direction for _, direction in solve([], (cur_pos, ''))[1:]]
    
    return result
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = []
    start = problem.getStartState()
    visited = set()
    visited.add(start)
    path = {}
    path[start] = ('', '')
    queue.append((start))
    find = False
    final = None
    while queue:
        # if find: break
        current = queue.pop(0)
        if problem.isGoalState(current):
            break
        for pos, direction, cost in problem.getSuccessors(current):
            if pos not in visited:
                queue.append(pos)
                visited.add(pos)
                path[pos] = (current, direction)
                # print(pos)
                # problem.getSuccessors(pos)
                if problem.isGoalState(pos):
                    find = True
                    final = pos
                if find: break

    curr = final
    res = []
    while curr != start:
        curr, direction = path[curr]
        res.append(direction)
    res.reverse()
    return res
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueueWithFunction
    def getPriority(item):
        return item[1]
    pQueue = PriorityQueueWithFunction(getPriority)

    start = problem.getStartState()
    visited = set()
    visited.add(start)
    path = {}
    path[start] = ('', '', 0)
    pQueue.push((start, 0))
    find = False
    final = None
    while not pQueue.isEmpty():
        print('--- P QUEUE ---')
        print(pQueue)
        # if find: break
        current, curr_cost = pQueue.pop()
        # print(current)
        if problem.isGoalState(current):
            break
        for pos, direction, cost in problem.getSuccessors(current):
            if pos in visited:
                if path[pos][2] > curr_cost + cost:
                    path[pos] = (current, direction, curr_cost+cost)
            else:
                pQueue.push((pos, cost + curr_cost))
                visited.add(pos)
                path[pos] = (current, direction, curr_cost+cost)
                if problem.isGoalState(pos):
                    final = pos
    curr = final
    res = []
    print('------------------------')
    for p in path:
        print(f'{p}: {path[p]}')
    while curr != start:
        curr, direction, cost = path[curr]
        res.append(direction)
    res.reverse()
    return res

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueueWithFunction
    from searchAgents import manhattanHeuristic

    # print("aqui")

    def getPriority(item):
        return item[1]
    pQueue = PriorityQueueWithFunction(getPriority)

    start = problem.getStartState()
    visited = set()
    visited.add(start)
    path = {}
    path[start] = ('', '', 0 + manhattanHeuristic(start, problem))
    pQueue.push((start, 0 + manhattanHeuristic(start, problem)))
    find = False
    final = None
    while not pQueue.isEmpty():
        # print('--- P QUEUE ---')
        # print(pQueue)
        # if find: break
        current, curr_cost = pQueue.pop()
        print(problem.goal)
        # print(current)
        if problem.isGoalState(current):
            break
        for pos, direction, cost in problem.getSuccessors(current):
            if pos in visited:
                if path[pos][2] > curr_cost + cost + manhattanHeuristic(pos, problem):
                    path[pos] = (current, direction, curr_cost+cost+manhattanHeuristic(pos, problem))
            else:
                pQueue.push((pos, cost + curr_cost+manhattanHeuristic(pos, problem)))
                visited.add(pos)
                path[pos] = (current, direction, curr_cost+cost+manhattanHeuristic(pos, problem))
                if problem.isGoalState(pos):
                    final = pos
    curr = final
    res = []
    print('------------------------')
    # for p in path:
        # print(f'{p}: {path[p]}')
    while curr != start:
        curr, direction, cost = path[curr]
        res.append(direction)
    res.reverse()
    return res


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
