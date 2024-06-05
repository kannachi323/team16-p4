import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        
        # *** Your Code Here ***
        
        # try to get to oldFood
        foodDistances = []
        for food in newFood.asList():
            mdist = distance.manhattan(newPosition, food)
            foodDistances.append(mdist)
        
        # food should be available
        minFoodDistance = 0
        if len(foodDistances) > 0:
            minFoodDistance = 1.0 / min(foodDistances)
        
        ghostPenalty = 0
        for i, ghost in enumerate(newGhostStates):
            ghostDistance = distance.manhattan(newPosition, ghost.getPosition())
            if newScaredTimes[i] == 0 and ghostDistance < 2:
                ghostPenalty += 200
        
        # pacman should not be stopping
        stopPenalty = 0
        if action == Directions.STOP:
            stopPenalty = 100
        
        return successorGameState.getScore() - ghostPenalty + minFoodDistance - stopPenalty
    
class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getLegalActionsExceptStop(self, gameState, agent):
        actions = gameState.getLegalActions(agent)
        return [action for action in actions if action != Directions.STOP]

    def maxValue(self, gameState, agent, depth):
        PACMAN = 0
        if depth == self.getTreeDepth() or gameState.isOver():
            return self.getEvaluationFunction()(gameState)
        value = float('-inf')
        actions = self.getLegalActionsExceptStop(gameState, agent)
        for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            nextAgent = agent + 1
            if agent + 1 >= gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1
                value = max(value, self.minValue(nextState, nextAgent, nextDepth))
            else:
                value = max(value, self.minValue(nextState, nextAgent, depth))
        return value
    
    def minValue(self, gameState, agent, depth):
        if depth == self.getTreeDepth() or gameState.isOver():
            return self.getEvaluationFunction()(gameState)
        value = float('inf')
        actions = self.getLegalActionsExceptStop(gameState, agent)
        if agent == gameState.getNumAgents() - 1:
            for action in actions:
                nextState = gameState.generateSuccessor(agent, action)
                value = min(value, self.maxValue(nextState, 0, depth + 1))
        else:
            for action in actions:
                nextState = gameState.generateSuccessor(agent, action)
                value = min(value, self.minValue(nextState, agent + 1, depth))
        
        return value
        
    def getAction(self, gameState):
        actions = self.getLegalActionsExceptStop(gameState, 0)
        bestValue = float('-inf')
        bestAction = None

        for action in actions:
            currValue = self.minValue(gameState.generateSuccessor(0, action), 1, 0)
            if currValue > bestValue:
                bestValue = currValue
                bestAction = action
        
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def getLegalActionsExceptStop(self, gameState, agent):
        actions = gameState.getLegalActions(agent)
        return [action for action in actions if action != Directions.STOP]

    def maxValue(self, gameState, alpha, beta, agent, depth):
        PACMAN = 0
        if depth == self.getTreeDepth() or gameState.isOver():
            return self.getEvaluationFunction()(gameState)
        value = float('-inf')
        actions = self.getLegalActionsExceptStop(gameState, agent)
        for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            nextAgent = agent + 1
            if nextAgent >= gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1
                value = max(value, self.minValue(nextState, alpha, beta, nextAgent, nextDepth))
            else:
                value = max(value, self.minValue(nextState, alpha, beta, nextAgent, depth))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value
    
    def minValue(self, gameState, alpha, beta, depth, agent):
        if depth == self.getTreeDepth() or gameState.isOver():
            return self.getEvaluationFunction()(gameState)
        value = float('inf')
        actions = self.getLegalActionsExceptStop(gameState, agent)
        if agent == gameState.getNumAgents() - 1:
            for action in actions:
                nextState = gameState.generateSuccessor(agent, action)
                value = min(value, self.maxValue(nextState, alpha, beta, 0, depth + 1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
        else:
            for action in actions:
                nextState = gameState.generateSuccessor(agent, action)
                value = min(value, self.minValue(nextState, alpha, beta, agent + 1, depth))
                if value <= alpha:
                    return value
                beta = min(beta, value)
        
        return value
        
    def getAction(self, gameState):
        actions = self.getLegalActionsExceptStop(gameState, 0)
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None

        for action in actions:
            currValue = self.minValue(gameState.generateSuccessor(0, action), alpha, beta, 1, 0)
            if currValue > bestValue:
                bestValue = currValue
                bestAction = action
            alpha = max(alpha, bestValue)
        
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getLegalActionsExceptStop(self, gameState, agent):
        actions = gameState.getLegalActions(agent)
        return [action for action in actions if action != Directions.STOP]

    def maxValue(self, gameState, agent, depth):
        PACMAN = 0
        if depth == self.getTreeDepth() or gameState.isOver():
            return self.getEvaluationFunction()(gameState)
        value = float('-inf')
        actions = self.getLegalActionsExceptStop(gameState, agent)
        if agent == PACMAN:
            for action in actions:
                nextState = gameState.generateSuccessor(agent, action)
                value = max(value, self.expectedValue(nextState, agent + 1, depth))
            return value
    
    def expectedValue(self, gameState, agent, depth):
        if depth == self.getTreeDepth() or gameState.isOver():
            return self.getEvaluationFunction()(gameState)
        totalValue = 0
        actions = self.getLegalActionsExceptStop(gameState, agent)
        if len(actions) == 0:
            return self.getEvaluationFunction()(gameState)
        for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:
                totalValue += self.maxValue(nextState, 0, depth + 1) / len(actions)
            else:
                totalValue += self.expectedValue(nextState, agent + 1, depth) / len(actions)
        return totalValue
        
    def getAction(self, gameState):
        actions = self.getLegalActionsExceptStop(gameState, 0)
        bestValue = float('-inf')
        bestAction = None

        for action in actions:
            currValue = self.expectedValue(gameState.generateSuccessor(0, action), 1, 0)
            if currValue > bestValue:
                bestValue = currValue
                bestAction = action
        
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>

    Similar to the first betterEvaluationFunction, we must try to penalize/reward PACMAN. However,
    this time we wiwll only focus on game states such ass
    1. distance to nearest food (closer is better) -> that's why I use inverse
    2. number of remaining food pellets (fewer is better)
    3. distance to scared ghost and brave ghosts (closer is worse)
    4. remaining capsule powerups and distances to them (fewer is better)
    5. game score at a certatin state (higher is better)

    For score adjustments, I kept plugging in different numbers until I found values
    that have a higher win rate.

    """
    # always start with current game state score
    score = currentGameState.getScore()

    newPosition = currentGameState.getPacmanPosition()

    foodList = currentGameState.getFood().asList()
    foodDistances = []

    for food in foodList:
        mdist = distance.manhattan(newPosition, food)
        foodDistances.append(mdist)
    
    nearestFoodDistance = 0
    if foodList:
        nearestFoodDistance = min(foodDistances)

    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.getScaredTimer() for ghostState in ghostStates]
    braveGhosts = []
    scaredGhosts = []
    for i, ghost in enumerate(ghostStates):
        if scaredTimes[i] == 0:
            braveGhosts.append(ghost)
        elif scaredTimes[i] > 0:
            scaredGhosts.append(ghost)
    
    braveGhostDistance = float('inf')
    braveGhostDistances = []
    if braveGhosts:
        for ghost in braveGhosts:
            braveGhostDistances.append(distance.manhattan(newPosition, ghost.getPosition()))
        braveGhostDistance = min(braveGhostDistances)

    scaredGhostDistance = float('inf')
    scaredGhostDistances = []
    if scaredGhosts:
        for ghost in scaredGhosts:
            scaredGhostDistances.append(distance.manhattan(newPosition, ghost.getPosition()))
        scaredGhostDistance = min(scaredGhostDistances)

    # adjusting scores based on penalites/rewards
    score += max(10 - nearestFoodDistance, 0)
    if braveGhostDistance < 2:
        score -= 999
    
    # we should try to chase the ghost if they are scared
    elif scaredGhostDistance < 2:
        score += max(10 * scaredGhostDistance, 0)

    score -= len(foodList)

    capsules = currentGameState.getCapsules()
    score -= len(capsules)

    return score
class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)