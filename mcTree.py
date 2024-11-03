import math,random
class Node:
    def __init__(self,game_state,parent):
        self.state = game_state
        self.parent = parent
        self.children = {}#{action: child node}
        self.visits = 0
        self.reward = 0
        self.fullyExpand = False#所有action都已经加入children
    #运行时间越长，reward越大，没考虑事物
    #运行时间越长，消耗健康值越高，这个也没考虑
    def getReward(self):
        if(self.endState()):
            return int(self.state['turn'])
        else:
            return 10000
        
    def endState(self):
        if(len(self.getNextActions())==0 or self.state['you']['health']==0):
            return True
        return False
    #a：action
    #返回在这个node perform这个action后的下一个node
    def performAction(self,a):
        from copy import deepcopy
        nextState = deepcopy(self.state)
        #改头的位置
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
        #检查是否吃掉棋盘上的事物
        if(nextState['you']['head'] in nextState['board']['food']):
            nextState['board']['food'].remove(nextState['you']['head'])
            nextState['you']['health']=100
        else:
            nextState['you']['health']-=1
        #去掉尾巴
        nextState['you']['body'].pop()
        nextState['you']['body'].insert(0,nextState['you']['head'])
        #snakes应该是棋盘上所有蛇的合集，把自己的蛇更新一下
        for q in range(len(nextState['board']['snakes'])):
            if(nextState['board']['snakes'][q]['id']==nextState['you']['id']):
                nextState['board']['snakes'][q]=deepcopy(nextState['you'])
                break  
        return Node(nextState,self)
            
    #下一步所有可能的action，无筛选
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
            
        left = {'x': self.state['you']['head']['x'] -1, 'y': self.state['you']['head']['y']}
        right = {'x':self.state['you']['head']['x'] +1, 'y': self.state['you']['head']['y']}
        up = {'x': self.state['you']['head']['x'], 'y': self.state['you']['head']['y'] + 1}
        down = {'x': self.state['you']['head']['x'], 'y': self.state['you']['head']['y'] - 1}
            
        bodyList=list()
        for i in range(len(self.state['board']['snakes'])):
            bodyList.extend(self.state['board']['snakes'][i]['body'])

        if (left in bodyList):
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
    #每次移动产生一个root，从这个root开始search，返回的action就是下一步move
    def search(self,start):
        import time
        self.root = Node(start,None)
        timeLimit = time.time() + self.maxTime
        count = 0
        self.maxIter=5
        while(time.time()<timeLimit and count <self.maxIter):
            count+=1
            node = self.select(self.root)
            t = node
            c = 0
            #t是我们目前选出的下一个node，在t随机进行一个action，直到endState或者循环>10停止搜索（dfs）
            while(t.endState()==False and c<10):
                c+=1
                t = t.performAction(random.choice(t.getNextActions()))
            #t在搜索的最后一个节点，算reward，把通往t的路径上每一个node的reward更新（backProp）
            reward = t.getReward()
            self.backProp(t,reward)
        #从root的child node里面挑出一个reward最高的，返回对应的action，这个是最终结果
        bestC = self.getBestChild(self.root, 0)
        action = None
        for nextA in self.root.children.keys():
            if(self.root.children[nextA]==bestC):
                action=nextA
        #如果他返回0就是下一步没有可行的action了。。。。。。
        print(len(self.root.getNextActions()))
        return action
    
    #如果所有children都已经被展开过了，挑reward值最大的那个路径的尽头的node返回，我们接着这个node继续dfs，更新reward
    #如果没有，随便一个action，下一个node展开（就是dfs）
    def select(self, node):
        while(node.endState()==False):
            if(node.fullyExpand):
                node = self.getBestChild(node)
            else:
                return self.expand(node)#next node
        return node
    #这里node应该是dfs搜索的尽头，从root到node路径上的每个节点reward更新
    def backProp(self,node,reward):
        while(node):
            node.visits+=1
            node.reward+=reward
            node= node.parent   
        
    #选择一个reward比较高的node，v是一个threshold，大于v的随机返回，如果没有大于v的返回最大的
    def getBestChild(self,node,exploration=0):
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
   
        if(len(bn)==0):
            v=0
            curNode=None
            for n in node.children.values():
                if(n.reward>v):
                    v=n.reward
                    curNode=n
            return curNode
        choice = random.choice(bn)
        position = bn.index(choice)  # 获取元素在列表中的索引
        # print(f"选中的元素位于列表的第 {position + 1} 位。一共{len(bn)}个元素可以选")
        return choice
    
    #对于node，找到第一个没有被dfs探索过的node
    def expand(self,node):
        actions = node.getNextActions()
        for a in actions:
            if a not in node.children:
                node.children[a] = node.performAction(a)
                if(len(actions)==len(node.children)):
                    node.fullyExpand = True
                return node.children[a]
            
        print("Should never reach here")
                
        
        
        
        