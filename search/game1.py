
'''
Start with a number N.
Players take turns either decrementing N or replacing with ceil(N/2)
The player that is left with 0 wins
'''
class HalvingGame():
    def __init__(self, N):
        self.N = N
    
    def startState(self):
        return (+1, self.N)
    
    def isEnd(self, state):
        player, num = state
        return num == 0
    
    def utility(self, state):
        player, num = state
        assert num == 0
        return player * float('inf')
    
    def actions(self, state):
        return ['-', '/']
    
    def player(self, state):
        player, num = state
        return player
    
    def succ(self, state, action):
        
        player, num = state
        if action == '-':
            return (-player, num-1)
        elif action == '/':
            return (-player, num//2)
        

def minimaxPolicy(game, state):
    def recurse(state):
        if game.isEnd(state):
            return (game.utility(state), 'none')
        choices = [
            (recurse(game.succ(state,action))[0], action)
            for action in game.actions(state)
        ]
        if game.player(state) == +1:
            return max(choices)
        elif game.player(state) == -1:
            return min(choices)
        
    value, action = recurse(state)
    print('minimax says action = {}, value = {}'.format(action, value))
    return action
        
        
def humanPolicy(game, state):
    while True:
        action = input('input action:')
        if action in game.actions(state):
            return action
        

# policies = {+1: humanPolicy, -1: humanPolicy}
# policies = {+1: humanPolicy, -1: minimaxPolicy}
policies = {+1: minimaxPolicy, -1: minimaxPolicy}
game = HalvingGame(N=15)
state = game.startState()

while not game.isEnd(state):
    print('='*10, state)
    player = game.player(state)
    policy = policies[player]
    action = policy(game, state)
    state = game.succ(state,action)
    

print('utility = {}'.format(game.utility(state)))
    