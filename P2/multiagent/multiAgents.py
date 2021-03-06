# Andrew Katsanevas
# Bradley Dawn
# CS 4300 Project 2

# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood();
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        evaluation = 0

        # Change evaluation based on remaining food
        for x in range(currentFood.width):
          for y in range(currentFood.height):
            if currentFood[x][y]:
              foodDist = manhattanDistance((x,y), newPos)
              # Getting food is good
              if foodDist == 0:
                evaluation += 10
              # Being close to food is also good
              else:
                evaluation += 1.0 / foodDist

        # Being close to ghosts is bad.
        for ghost in newGhostStates:
          ghostDist = manhattanDistance(ghost.getPosition(), newPos)
          if ghostDist < 2 and ghost.scaredTimer == 0:
              evaluation = float('-inf')
        
        return evaluation

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.maxMove(gameState, 0, 0)[1]

    # Find the max score move of min score moves
    def maxMove(self, gameState, currentDepth, agent):
    	actions = gameState.getLegalActions(agent)

    	# The game is over
    	if not actions or gameState.isWin() or currentDepth >= self.depth:
    		return self.evaluationFunction(gameState), Directions.STOP

    	maxCost = float('-inf')
    	maxMove = Directions.STOP
    	for move in actions:
    		succ = gameState.generateSuccessor(agent, move)
    		nextCost = self.minMove(succ, currentDepth, agent+1)[0]
    		if nextCost > maxCost:
    			maxCost = nextCost
    			maxMove = move

    	return maxCost, maxMove

    # Find the min score move
    def minMove(self, gameState, currentDepth, agent):
    	actions = gameState.getLegalActions(agent)

    	# The game is over
    	if not actions or gameState.isLose() or currentDepth >= self.depth:
    		return self.evaluationFunction(gameState), Directions.STOP

    	minCost = float('inf')
    	minMove = Directions.STOP
    	for move in actions:
    		succ = gameState.generateSuccessor(agent, move)
    		nextCost = 0

    		# Go to the next depth
    		if agent == gameState.getNumAgents() - 1:
    			nextCost = self.maxMove(succ, currentDepth+1, 0)[0]
    		# Go to the next agent
    		else:
    			nextCost = self.minMove(succ, currentDepth, agent+1)[0]

    		if nextCost < minCost:
    			minCost = nextCost
    			minMove = move

    	return minCost, minMove



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.maxMove(gameState, 0, 0, float('-inf'), float('inf'))[1]

    # Find the max score move of min score moves
    def maxMove(self, gameState, currentDepth, agent, alpha, beta):
    	actions = gameState.getLegalActions(agent)

    	# The game is over
    	if not actions or gameState.isWin() or currentDepth >= self.depth:
    		return self.evaluationFunction(gameState), Directions.STOP

    	maxCost = float('-inf')
    	maxMove = Directions.STOP
    	for move in actions:
    		succ = gameState.generateSuccessor(agent, move)
    		nextCost = self.minMove(succ, currentDepth, agent+1, alpha, beta)[0]
    		if nextCost > maxCost:
    			maxCost = nextCost
    			maxMove = move

    		# Check beta
    		if maxCost > beta:
    			return maxCost, maxMove

    		# Update alpha
    		alpha = max(alpha, maxCost)

    	return maxCost, maxMove

    # Find the min score move
    def minMove(self, gameState, currentDepth, agent, alpha, beta):
    	actions = gameState.getLegalActions(agent)

    	# The game is over
    	if not actions or gameState.isLose() or currentDepth >= self.depth:
    		return self.evaluationFunction(gameState), Directions.STOP

    	minCost = float('inf')
    	minMove = Directions.STOP
    	for move in actions:
    		succ = gameState.generateSuccessor(agent, move)
    		nextCost = 0

    		# Go to the next depth
    		if agent == gameState.getNumAgents() - 1:
    			nextCost = self.maxMove(succ, currentDepth+1, 0, alpha, beta)[0]
    		# Go to the next agent
    		else:
    			nextCost = self.minMove(succ, currentDepth, agent+1, alpha, beta)[0]

    		if nextCost < minCost:
    			minCost = nextCost
    			minMove = move

    		# Check alpha
    		if minCost < alpha:
    			return minCost, minMove

    		# Update beta
    		beta = min(beta, minCost)

    	return minCost, minMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.maxMove(gameState, 0, 0)[1]

    # Find the max score move of average costs
    def maxMove(self, gameState, currentDepth, agent):
    	actions = gameState.getLegalActions(agent)

    	# The game is over
    	if not actions or gameState.isWin() or currentDepth >= self.depth:
    		return self.evaluationFunction(gameState), Directions.STOP

    	maxCost = float('-inf')
    	maxMove = Directions.STOP
    	for move in actions:
    		succ = gameState.generateSuccessor(agent, move)
    		nextCost = self.expectedCost(succ, currentDepth, agent+1)
    		if nextCost > maxCost:
    			maxCost = nextCost
    			maxMove = move

    	return maxCost, maxMove

    # Find the expected cost
    def expectedCost(self, gameState, currentDepth, agent):
    	actions = gameState.getLegalActions(agent)

    	# The game is over
    	if not actions or gameState.isLose() or currentDepth >= self.depth:
    		return self.evaluationFunction(gameState)

    	costs = []
    	for move in actions:
    		succ = gameState.generateSuccessor(agent, move)
    		nextCost = 0

    		# Go to the next depth
    		if agent == gameState.getNumAgents() - 1:
    			nextCost = self.maxMove(succ, currentDepth+1, 0)[0]
    		# Go to the next agent
    		else:
    			nextCost = self.expectedCost(succ, currentDepth, agent+1)
    		# Put this next cost into the list
    		costs.append(nextCost)

    	# Get the average cost
    	average = sum(costs) / float(len(costs))
    	return average

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Takes into account the current score, remaining food/capsules, 
      distance from nearest food, and distance from nearest ghost.
    """

    # A high score is probably good. We'll use it as a baseline.
    evaluation = currentGameState.getScore()

    # We really want to get the number of foods and capsules to be lower.
    evaluation -= 10 * (currentGameState.getNumFood() + len(currentGameState.getCapsules()))

    # Being far from food is bad.
    minFoodDist = float('inf')
    subtract = 0
    for food in currentGameState.getFood().asList():
    	foodDist = util.manhattanDistance(currentGameState.getPacmanPosition(), food)
    	if foodDist < minFoodDist:
    		minFoodDist = foodDist
    		subtract = foodDist
    evaluation -= subtract

    # Being close to a ghost is bad. Let's avoid them no matter what.
    for ghost in currentGameState.getGhostStates():
    	ghostDist = util.manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition())
    	if ghostDist < 2:
    		evaluation = float('-inf')

    return evaluation

# Abbreviation
better = betterEvaluationFunction

