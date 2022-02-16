import numpy as np
from math import pi
import turtle as tt
import matplotlib.cm as cm
from tqdm import tqdm

class Map():
    def __init__(self,map_width,map_height,types,populations,max_speeds,max_accs,rules,rule_probs,eps=1e-7):
        self.map_width,self.map_height=map_width,map_height # width and height of the map
        self.types=types # types of objects
        self.populations=populations # populations of different types
        self.max_speeds=max_speeds # max velocities of different types
        self.max_accs=max_accs # max accelerations of different types
        self.rules=rules # rules of attraction
        self.rule_probs=rule_probs # probabilities of attraction if within range
        self.positions=dict() # dict of arrays of different types of objects
        self.speeds=dict() # dict os arrays of different types of objects
        self.distances=dict() # dict of pairwise distances for all rules
        self.eps=eps # small number to avoid division by zero

        self.init_objects()
        self.init_GUI()
    
    def random_unit_vectors(self,num):
        # random sampling of speeds
        thetas=np.random.rand(num,1)*2*pi
        return np.concatenate([np.cos(thetas),np.sin(thetas)],axis=1)
    
    # Initialize positions and speeds for all types, compute the pairwise distances
    def init_objects(self):
        self.positions.clear()
        for type in self.types:
            population=self.populations[type]
            # random sampling of positions 
            self.positions[type]=np.random.rand(population,2)*(self.map_width,self.map_height)
            # random sampling of speeds
            self.speeds[type]=self.max_speeds[type]*np.random.rand(population,1)*self.random_unit_vectors(population)
        self.update_distances()
    
    # Compute pairwise distances of all rules
    def update_distances(self):
        for rule in self.rules.keys():
            type1,type2=rule
            pos1,pos2=self.positions[type1],self.positions[type2]
            self.distances[rule]=np.sqrt(-2*pos1.dot(pos2.T)+np.sum(pos1**2,axis=1,keepdims=True)+np.sum(pos2**2,axis=1))

    # Do one iteration and return a reference to self.positions
    def iterate(self):
    
        def randomIndex(arr):
            idx=np.where(arr)[0]
            if len(idx)==0: return -1
            return np.random.choice(idx)
    
        # assuming objects can only be attracted by at most one type. This will be improved later.
        updated_types=list()

        # update speed
        for rule,dist in self.rules.items():
            type1,type2=rule
            if type1 in updated_types: continue

            obj1_pos,obj2_pos=self.positions[type1],self.positions[type2]
            num_type1=self.populations[type1]

            # get random index of type2 that attracts type1, return -1 if not exists 
            attracted_indices=np.apply_along_axis(randomIndex,1,self.distances[rule]<=dist)

            # if some type2 is within range, attracted with probability rule_probs[type1]
            attracted_mask=(attracted_indices!=-1)*(np.random.rand(num_type1)<self.rule_probs[rule])
            attracted_indices=attracted_indices[attracted_mask]
            
            # if attracted, accelerate towards type2 with max acceleration
            accs=obj2_pos[attracted_indices]-obj1_pos[attracted_mask]
            accs=self.max_accs[type1]*accs/(np.linalg.norm(accs,axis=1,keepdims=True)+self.eps)

            # update speed by adding acceleration
            self.speeds[type1][attracted_mask]+=accs
            num_not_att=np.sum(attracted_mask==False)
            self.speeds[type1][attracted_mask==False]+=self.max_accs[type1]*np.random.rand(num_not_att,1)*self.random_unit_vectors(num_not_att)

            # speed is clipped to max speed
            self.speeds[type1]=self.max_speeds[type1]*self.speeds[type1] \
                /np.maximum(self.max_speeds[type1],np.linalg.norm(self.speeds[type1],axis=1,keepdims=True)+self.eps)

            updated_types.append(type1)
        
        # update unattracted types
        for type in self.types:
            if type not in updated_types:
                num_type=self.populations[type]
                # update speed by adding acceleration
                self.speeds[type]+=self.max_accs[type]*np.random.rand(num_type,1)*self.random_unit_vectors(num_type)

                # speed is clipped to max speed
                self.speeds[type]=self.max_speeds[type]*self.speeds[type] \
                    /np.maximum(self.max_speeds[type],np.linalg.norm(self.speeds[type],axis=1,keepdims=True)+self.eps)
            
        # update positions
        for type,s in self.speeds.items():
            
            pos=self.positions[type]
            pos+=s

            # reverse the direction of speed when collision happens
            mask=~((pos[:,0]>=0)*(pos[:,0]<=self.map_width))
            s[mask,0]=-s[mask,0]
            mask=~((pos[:,1]>=0)*(pos[:,1]<=self.map_height))
            s[mask,1]=-s[mask,1]

            # clip position to value within map
            
            pos[:,0]+=2*(np.maximum(-pos[:,0],0)-np.maximum(pos[:,0]-self.map_width,0))
            pos[:,1]+=2*(np.maximum(-pos[:,1],0)-np.maximum(pos[:,1]-self.map_height,0))
        
        # update distances
        self.update_distances()

        return self.positions

    def init_GUI(self):
        tt.tracer(False)
        tt.setworldcoordinates(0,0,self.map_width,self.map_height*1.1)
        tt.colormode(1.0)
        tt.penup()
        tt.shapesize(0.3,0.3,0.3)
        tt.shape(name='turtle')
        self.colors=cm.rainbow(np.linspace(0,1,len(types)))[:,:3]
        self.textTurtle=tt.Turtle()
        
        self.textTurtle.goto(0,0)
        self.textTurtle.setheading(0)
        self.textTurtle.pendown()
        self.textTurtle.color('black')
        self.textTurtle.shape('circle')
        self.textTurtle.shapesize(0.3,0.3,0.3)
        for i in range(4):
            if i%2==0: self.textTurtle.fd(self.map_width)
            else: self.textTurtle.fd(self.map_height)
            self.textTurtle.left(90)
        self.textTurtle.penup()
        self.textTurtle.goto(0,self.map_height*1.05)
        self.textTurtle.setheading(0)
        for type,color in zip(self.types,self.colors):
            self.textTurtle.color(*color)
            self.textTurtle.stamp()
            self.textTurtle.write(type)
            self.textTurtle.fd(0.5*map_width/max(len(self.types),5))
        self.textTurtle.ht()
        tt.update()
    
    def update_GUI(self):
        positions=map.iterate()
        tt.clear()
        for i,type in enumerate(positions.keys()):
            pos=positions[type]
            tt.color(*self.colors[i])
            for p in pos:
                tt.goto(p[0],p[1])
                tt.stamp()
        tt.update()

if __name__=='__main__':
    # Initialization of map

    map_width,map_height=1000,800 # width and height of the map
    types=['A','B','C'] # A,B,C types of objects
    populations={types[i]:v for i,v in enumerate([100,110,90])} # populations of different types
    max_speeds={types[i]:v for i,v in enumerate([3,4,5])} # max velocities of different types
    max_accs={types[i]:v for i,v in enumerate([0.5,0.5,0.5])} # max accelerations of different types
    rules={('A','B'):20,('B','C'):20} # B attracts A (B->A) within dist of 8, C attracts B (C->B) within dist of 10
    rule_probs={('A','B'):0.8,('B','C'):0.7} # probabilities of attraction if within range

    map=Map(map_width,map_height,types,populations,max_speeds,max_accs,rules,rule_probs)
    n_iters=1000
    for rule in rules.keys():
        type1,type2=rule
        print('%s is attracted by %s'%(type1,type2))
    map.init_GUI()
    for i in tqdm(range(n_iters)):
        map.update_GUI()
    tt.done()

