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
import math

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

        "Add more of your code here if you want to"

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        '''
        print("successorgamestate ",successorGameState)
        print("newPos ",newPos)
        print("newfood ",newFood)
        print("newghoststates", newGhostStates)
        print("newscaredtimes ",newScaredTimes)
        print("score",successorGameState.getScore())
        '''
        distancetofoods=[]
        for food in newFood:
            distancetofoods.append(manhattanDistance(newPos,food))

        distancetoghosts=[]
        for ghost in successorGameState.getGhostPositions():
            distancetoghosts.append(manhattanDistance(ghost, newPos))

        if min(newScaredTimes)==0:
            ghostdist=-3 / (min(distancetoghosts) + 1)
        else:
            ghostdist=0.7 / (min(distancetoghosts) + 1)


        if(len(distancetofoods)==0):
            distancetoclosestfood=1;
        else:
            distancetoclosestfood= min(distancetofoods)+1
        numofremainingfood =len(newFood)


        return successorGameState.getScore()+0.5/(distancetoclosestfood+1)-numofremainingfood+ghostdist

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

    def isTerminal(self, state, depth):
        return self.depth == depth or state.isWin() or state.isLose()

    def maximize(self,state, depth):
        if self.isTerminal(state, depth):
            return self.evaluationFunction(state)
        else:
            nextStates=[]
            scores=[]
            for act in state.getLegalActions(0):
                nextStates.append(state.generateSuccessor(0, act))

            for move in nextStates:
                scores.append(self.minimize(move, 1, depth))

            return max(scores)

    def minimize(self,state,index, depth):
        if self.isTerminal(state, depth):
            return self.evaluationFunction(state)
        else:
            nextStates=[]
            scores=[]
            for act in state.getLegalActions(index):
                nextStates.append(state.generateSuccessor(index, act))

            agents = state.getNumAgents()

            if index < agents - 1:  # check if this is the last ghost's turn
                for move in nextStates:
                    scores.append(self.minimize(move, index + 1, depth))

            else:  # otherwise it is Pacman's turn so we maximize
                for move in nextStates:
                    scores.append(self.maximize(move, depth + 1))
            return min(scores)

    def minimax(self,gameState):
        actions = gameState.getLegalActions(0)
        successors=[]
        scores=[]

        for act in actions:
            successors.append(gameState.generateSuccessor(0, act))
        for state in successors:
            scores.append(self.minimize(state, 1, 0))

        bestMove = []
        for index in range(len(scores)):
            if scores[index] == max(scores):
                bestMove.append(index)
            index += 1
        return actions[random.choice(bestMove)]

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
        "*** YOUR CODE HERE ***"

        return self.minimax(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    	Your minimax agent with alpha-beta pruning (question 3)
    """

    def minimizer(self,state, depth, agent, alpha, beta):
        if agent == state.getNumAgents():
            return self.maximizer(state, depth + 1, alpha, beta)

        val = None
        for action in state.getLegalActions(agent):
            successor = self.minimizer(state.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)

            if val is None:
                val = successor
            else:
                val = min(val, successor)
            if alpha is not None and val < alpha:
                return val
            if beta is None:
                beta = val
            else:
                min(beta, val)

        if val is None:
            return self.evaluationFunction(state)

        return val

    def maximizer(self,state, depth, alpha, beta):

        if depth > self.depth:
            return self.evaluationFunction(state)

        val = None
        for action in state.getLegalActions(0):
            successor = self.minimizer(state.generateSuccessor(0, action), depth, 1, alpha, beta)
            val = max(val, successor)

            if beta is not None and val > beta:
                return val

            alpha = max(alpha, val)

        if val is None:
            return self.evaluationFunction(state)

        return val

    def alphabeta(self,state):
        val, alpha, beta, best = None, None, None, None
        for action in state.getLegalActions(0):
            val = max(val, self.minimizer(state.generateSuccessor(0, action), 1, 1, alpha, beta))

            if alpha is None:
                alpha, best = val, action
            else:
                alpha, best = max(val, alpha), action if val > alpha else best

        return best

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        return self.alphabeta(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def isTerminal(self, state, depth):
        return self.depth == depth or state.isWin() or state.isLose()

    def expected(self, depth, state, index):
        if self.isTerminal(state,depth):
            return self.evaluationFunction(state)
        else:
            agents = state.getNumAgents()
            nextStates=[]
            scores = []
            for action in state.getLegalActions(index):
                nextStates.append(state.generateSuccessor(index, action))
            if index >= agents - 1:
                for state in nextStates:
                    scores.append(self.maximizer(depth + 1, state))
            else:
                for state in nextStates:
                    scores.append(self.expected(depth, state, index + 1))

            return sum(scores) / len(scores)

    def maximizer(self, depth, state):
        if self.isTerminal(state,depth):
            return self.evaluationFunction(state)
        nextStates = []
        scores = []
        for action in state.getLegalActions(0):
            nextStates.append(state.generateSuccessor(0, action))
        for state in nextStates:
            scores.append(self.expected(depth, state, 1))

        return max(scores)

    def expectimax(self, gameState):
        actions = gameState.getLegalActions(0)
        nextStates = []
        scores = []
        for action in gameState.getLegalActions(0):
            nextStates.append(gameState.generateSuccessor(0, action))
        for state in nextStates:
            scores.append(self.expected(0, state, 1))

        maxScore = max(scores)
        bestIndices = []
        for index in range(len(scores)):
            if scores[index] == maxScore:
                bestIndices.append(index)
        i = bestIndices[0]
        return actions[i]

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    distancetofoods=[]
    for food in newFood:
        distancetofoods.append(manhattanDistance(newPos,food))

    distancetoghosts=[]
    for ghost in currentGameState.getGhostPositions():
        distancetoghosts.append(manhattanDistance(ghost, newPos))

    if min(newScaredTimes)==0:
            ghostdist=-3 / (min(distancetoghosts) + 1)
    else:
            ghostdist=0.7 / (min(distancetoghosts) + 1)


    if(len(distancetofoods)==0):
            distancetoclosestfood=1;
    else:
            distancetoclosestfood= min(distancetofoods)+1
    numofremainingfood =len(newFood)


    return currentGameState.getScore()+0.5/(distancetoclosestfood+1)-numofremainingfood+ghostdist

# Abbreviation
better = betterEvaluationFunction

