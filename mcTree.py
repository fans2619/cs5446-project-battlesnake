class Node:
    def __init__(self,game_state,parent):
        self.state = game_state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.reward = 0
        self.fullyExpand = False
    def getReward(self):
        if(self.endState()):
            return self.turn
        else:
            return 10000

    def endState(self):
        if(len(self.getNextActions())==0 or self.state['you']['health']==0):
            return True
        return False
    def performAction(self,a):
        from copy import deepcopy
        nextState = deepcopy(self.state)
        if(a=='up'):
            nextState['you']['head']['y']+=1
        elif(a=='down'):
            nextState['you']['head']['y']-=1
        elif(a=='right'):
            nextState['you']['head']['x']+=1
        elif(a=='left'):
            nextState['you']['head']['x']-=1
        else:
            raise Exception(a)
  
        nextState['turn']+=1
        
        if(nextState['you']['head'] in nextState['board']['food']):
            nextState['board']['food'].remove(nextState['you']['head'])
            nextState['you']['health']=100
        else:
            nextState['you']['health']-=1
            
        nextState['you']['body'].pop()
        nextState['you']['body'].insert(0,nextState['you']['head'])
        
        for q in range(len(nextState['board']['snakes'])):
            if(nextState['board']['snakes'][q]['id']==nextState['you']['id']):
                nextState['board']['snakes'][q]=deepcopy(nextState['you'])
                break
            
        return Node(nextState,self)
            
        
    def getNextActions(self):
        res = ['up', 'down', 'left', 'right']
        if self.state['board']['width'] - self.state['you']['head']['x'] == 1:  
            res.remove('right')
        if self.state['you']['head']['x']== 0:                         
            res.remove('left')
        if self.state['board']['height'] -self.state['you']['head']['y'] == 1:  
            res.remove('up')
        if self.state['you']['head']['y'] == 0:                        
            res.remove('down')
        left = {'x': self.state['you']['head']['x'] - 1, 'y': self.state['you']['head']['y']}
        right = {'x':self.state['you']['head']['x'] + 1, 'y': self.state['you']['head']['y']}
        up = {'x': self.state['you']['head']['x'], 'y': self.state['you']['head']['y'] + 1}
        down = {'x': self.state['you']['head']['x'], 'y': self.state['you']['head']['y'] - 1}
        bodyList=list()
        for i in range(len(self.state['board']['snakes'])):
            bodyList.extend(self.state['board']['snakes'][i]['body'])

        if left in bodyList:
            res.remove('left')
        if right in bodyList:
            res.remove('right')
        if up in bodyList:
            res.remove('up')
        if down in bodyList:
            res.remove('down')
        return res
        

class mcTree:
    def __init__(self,maxTime, maxIter):
        self.maxTime = maxTime
        self.maxIter = maxIter
    def search(self,start):
        import time
        self.root = Node(start,None)
        timeLimit = time.time() + self.maxTime
        count = 0
        import random
        while(time.time()<timeLimit and count <self.maxIter):
            count+=1
            node = self.select(self.root)
            t = node
            
            while(t.endState()==False):
                t = node.performAction(random.choice(t.getNextActions())) 
            reward = t.getReward()
            
            self.backProp(node,reward)
            
        bestC = self.getBestChild(self.root, 0)
        action = (action for action, node in self.root.children.items()
                  if node is bestC).__next__()
        
        return action
            
    def select(self, node):
        if(node.endState()):
            return node
        while(node.endState()==False):
            if(node.fullyExpand):
                node = self.getBestChild(node)
            else:
                return self.expand(node)#next node
            
    def backProp(self,node,reward):
        while(node):
            node.visits+=1
            node.reward+=reward
            node= node.parent   
        
             
    def getBestChild(self,node,exploration):
        import math,random
        cons = 1/math.sqrt(2)
        bn = list()
        b = float("-inf")
        for n in node.children.values():
            v=n.reward/n.visits+cons*math.sqrt(2*math.log(node.visits)/n.visits)
            if(v>b):
                b=v
                bn =[n]
            elif(v==b):
                bn.append(n)
                
        return random.choice(bn)
        
    def expand(self,node):
        actions = node.getNextActions()
        for a in actions:
            if a not in node.children:
                node.children[a] = node.performAction(a)
                if(len(actions)==len(node.children)):
                    node.fullyExpand = True
                return node.children[a]
            
                
        
        
        
        