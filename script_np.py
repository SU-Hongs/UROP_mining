import numpy as np
from math import pi
import turtle as tt
import matplotlib.cm as cm
from tqdm import tqdm

class DistDict(dict):
    # Customized to avoid duplicated calculation of pairwise distances, like (A,B) and (B,A)...
    def __getitem__(self,key):
        if type(key)==tuple: new_key=tuple(sorted(list(key)))
        if key==new_key: return super().__getitem__(key)
        return super().__getitem__(new_key).T

    def __setitem__(self,key,val):
        if type(key)==tuple: new_key=tuple(sorted(list(key)))
        if key==new_key: return super().__setitem__(new_key,val)
        return super().__setitem__(new_key,val.T)

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
        self.distances=DistDict() # dict of pairwise distances for all rules
        self.eps=eps # small number to avoid division by zero

        self.init_objects()
        self.init_GUI()
    
    def random_unit_vectors(self,num):
        # random sampling of speeds
        thetas=np.random.rand(num,1)*2*pi
        return np.concatenate([np.cos(thetas),np.sin(thetas)],axis=1)
    
    def randomIndex(self,arr):
            idx=np.where(arr)[0]
            if len(idx)==0: return -1
            return np.random.choice(idx)
    
    # Initialize positions and speeds for all types, compute the pairwise distances
    def init_objects(self):
        self.positions.clear()
        for t in self.types:
            population=self.populations[t]
            # random sampling of positions 
            self.positions[t]=np.random.rand(population,2)*(self.map_width,self.map_height)
            # random sampling of speeds
            self.speeds[t]=self.max_speeds[t]*np.random.rand(population,1)*self.random_unit_vectors(population)
        # self.update_distances()
        self.compute_all_pairwise_dist()
    
    # Compute pairwise distances of all rules
    def update_distances(self):
        updated_pairs=list()
        for rule in self.rules.keys():
            type1,type2=rule
            if type(type2)==str and set(rule) not in updated_pairs: # attracted by single object
                pos1,pos2=self.positions[type1],self.positions[type2]
                self.distances[rule]=np.sqrt(-2*pos1.dot(pos2.T)+np.sum(pos1**2,axis=1,keepdims=True)+np.sum(pos2**2,axis=1))
                updated_pairs.append(set(rule))
            elif type(type2)==tuple: # attracted by a group of objects
                for t in type2:
                    if set([type1,t]) not in updated_pairs:
                        pos1,pos2=self.positions[type1],self.positions[t]
                        self.distances[(type1,t)]=np.sqrt(-2*pos1.dot(pos2.T)+np.sum(pos1**2,axis=1,keepdims=True)+np.sum(pos2**2,axis=1))
                        updated_pairs.append(set([type1,t]))
    
    # Compute all pairwise distances
    def compute_all_pairwise_dist(self):
        for t1 in self.types:
            for t2 in self.types:
                if t2<=t1: continue # lexicographical order
                pos1,pos2=self.positions[t1],self.positions[t2]
                self.distances[(t1,t2)]=np.sqrt(-2*pos1.dot(pos2.T)+np.sum(pos1**2,axis=1,keepdims=True)+np.sum(pos2**2,axis=1))
    
    # Flatten rule to list of types
    def flatten_rule(rule):
        obj1,objs=rule
        if type(objs)==str: return [obj1,objs]
        elif type(objs)==tuple: return list(objs)+[obj1,]
    
    # Update speed for objects that is attracted by single object
    def single_attraction(self,rule,dist):

        type1,type2=rule
        obj1_pos,obj2_pos=self.positions[type1],self.positions[type2]
        num_type1=self.populations[type1]

        # get random index of type2 that attracts type1, return -1 if not exists 
        attracted_indices=np.apply_along_axis(self.randomIndex,1,self.distances[rule]<=dist)

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
    
    # Update speed for objects that is attracted by a group of objects
    def multi_attraction(self,rule,dist):
        type1,type2=rule
        obj1_pos=self.positions[type1]
        num_type1=self.populations[type1]

        # get random index of type2 that attracts type1, return -1 if not exists 
        attracted_indices=np.concatenate([np.apply_along_axis(self.randomIndex,1,self.distances[(type1,t)]<=dist).reshape(-1,1) for t in type2],axis=1)
        
        # if some type2 is within range, attracted with probability rule_probs[type1]
        attracted_mask=(~np.any(attracted_indices==-1,axis=1))&(np.random.rand(num_type1)<self.rule_probs[rule])
        attracted_indices=attracted_indices[attracted_mask]
        
        # get the center pos of the group of objects that attract type1
        centers=[np.mean([self.positions[t][idx[i]] for i,t in enumerate(type2)],axis=0).reshape(1,-1) for idx in attracted_indices]
        
        obj2_pos,accs=None,None
        if len(centers)>0: # there exists some attracted objects
            obj2_pos=np.concatenate(centers,axis=0)

            # if attracted, accelerate towards type2 with max acceleration
            accs=obj2_pos-obj1_pos[attracted_mask]
            accs=self.max_accs[type1]*accs/(np.linalg.norm(accs,axis=1,keepdims=True)+self.eps)

            # update speed by adding acceleration
            self.speeds[type1][attracted_mask]+=accs
        
        # randomly choose acceleration to update speed of non-attracted objects
        num_not_att=np.sum(attracted_mask==False)
        self.speeds[type1][attracted_mask==False]+=self.max_accs[type1]*np.random.rand(num_not_att,1)*self.random_unit_vectors(num_not_att)

        # speed is clipped to max speed
        self.speeds[type1]=self.max_speeds[type1]*self.speeds[type1] \
            /np.maximum(self.max_speeds[type1],np.linalg.norm(self.speeds[type1],axis=1,keepdims=True)+self.eps)
    
    # Update speed by rules
    def update_speeds(self):
    
        # assuming objects can only be attracted by at most one type. This will be improved later.
        updated_types=list()

        # update speed by rule
        for rule,dist in self.rules.items():
            if rule[0] in updated_types: continue
            if type(rule[1])==str: self.single_attraction(rule,dist)
            elif type(rule[1])==tuple: self.multi_attraction(rule,dist)
            updated_types.append(rule[0])
        
        # randomly update speed of unattracted types
        for t in self.types:
            if t not in updated_types:
                num_type=self.populations[t]
                # update speed by adding acceleration
                self.speeds[t]+=self.max_accs[t]*np.random.rand(num_type,1)*self.random_unit_vectors(num_type)

                # speed is clipped to max speed
                self.speeds[t]=self.max_speeds[t]*self.speeds[t] \
                    /np.maximum(self.max_speeds[t],np.linalg.norm(self.speeds[t],axis=1,keepdims=True)+self.eps)

    # Add speed to postition for each object, update distances
    def update_positions(self):
        # update positions
        for t,s in self.speeds.items():
            
            pos=self.positions[t]
            pos+=s

            # reverse the direction of speed when collision happens
            mask=~((pos[:,0]>=0)*(pos[:,0]<=self.map_width))
            s[mask,0]=-s[mask,0]
            mask=~((pos[:,1]>=0)*(pos[:,1]<=self.map_height))
            s[mask,1]=-s[mask,1]

            # find the position after reflection
            
            pos[:,0]+=2*(np.maximum(-pos[:,0],0)-np.maximum(pos[:,0]-self.map_width,0))
            pos[:,1]+=2*(np.maximum(-pos[:,1],0)-np.maximum(pos[:,1]-self.map_height,0))
        
        # update distances
        # self.update_distances()
        self.compute_all_pairwise_dist()

    # Do one iteration and return a reference to self.positions
    def iterate(self):
        self.update_speeds()
        self.update_positions()
        return self.positions

    def init_GUI(self):
        tt.tracer(False)
        tt.setworldcoordinates(0,0,self.map_width,self.map_height*1.1)
        tt.colormode(1.0)
        tt.penup()
        tt.shapesize(0.3,0.3,0.3)
        tt.shape(name='circle')
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
        for t,color in zip(self.types,self.colors):
            self.textTurtle.color(*color)
            self.textTurtle.stamp()
            self.textTurtle.write(t)
            self.textTurtle.fd(0.5*map_width/max(len(self.types),5))
        self.textTurtle.ht()
        tt.update()
    
    def update_GUI(self):
        positions=map.iterate()
        tt.clear()
        for i,t in enumerate(positions.keys()):
            pos=positions[t]
            tt.color(*self.colors[i])
            for p in pos:
                tt.goto(p[0],p[1])
                tt.stamp()
        tt.update()

    # idx of current active objects in the sub-region
    def compute_colocation(self, thres,rule,idx):
        obj1,objs = rule
        col_list = []
        for obj in objs:
            curr_dist = self.distances[(obj1,obj)]
            select_dist = curr_dist[idx]
            select_dist = (select_dist<thres).astype(int)
            select_dist = select_dist.sum(axis=1)
            select_dist = np.where(select_dist>0)[0]
            col_list.append( select_dist )
        final_col=[]
        for i in range(len(col_list)):
            # if i==len(col_list)-1: break
            if i==0: final_col = col_list[0]
            final_col = np.intersect1d(final_col,col_list[i])
        return len(final_col)


    # a function to select the objects in the targeted sub-region
    # return a dictionary of the the colocation density of the colocation patterns 
    # mode 1 stands for computing density for rules (i.e. at least a colocation pattern of size 2)
    # mode 2 stands for computing density for individual types of objects
    def compute_density(self, thres, mode):
        # first select the index of the objects that is in the range of the targeted subregion
        # define the left bottom corner of the sub-region and the length
        lb_x = int(self.map_width/4)
        lb_y = int(self.map_height/4)
        length = int(np.minimum(self.map_height,self.map_width)/2)
        area = (length**2)/10000 # divided by 100 to make the density moderate
        idx_dict = dict()
        for t in self.types:
            x, y = self.positions[t][:,0], self.positions[t][:,1]
            idx = np.where((x>=lb_x) & (x<lb_x+length) & (y>=lb_y) & (y<lb_y+length))[0]
            idx_dict[t] = idx

        density_dict = dict()
        # for mode 1
        if mode == 1:
            for rule in self.rules:
                t, _=rule
                idx = idx_dict[t]
                # select_dist = curr_dist[idx]
                # select_dist = (select_dist<thres).astype(int)
                # select_dist = select_dist.sum(axis=1)
                # select_dist = np.where(select_dist>0,1,0)
                # num_colo = len(select_dist)
                num_colo = self.compute_colocation(thres,rule,idx)
                density_dict[rule] = num_colo


        # for mode 2
        elif mode ==2:
            # in this case # of colocation patterns is just # of objects in the subregion
            for t in self.types:
                num_colo = len(idx_dict[t])
                density_dict[t] = num_colo
        
        return density_dict

if __name__=='__main__':
    # Initialization of map

    map_width,map_height=900,750 # width and height of the map
    types=['A','B','C','D'] # types of objects
    populations={types[i]:v for i,v in enumerate([210,150,150,150])} # populations of different types
    max_speeds={types[i]:v for i,v in enumerate([6,3,3,4])} # max velocities of different types
    max_accs={types[i]:v for i,v in enumerate([0.5,0.5,0.5,0.6])} # max accelerations of different types
    rule_list=[('A','B'),('C',('A','B')),('D',('A','B','C'))] # list of rules where the first is attracted by the second (e.g. (A,B) means A->B)
    #rule_list=[('A','B'),('C',('A','B')),('D','E'),('F',('D','E'))] # list of rules where the first is attracted by the second (e.g. (A,B) means A->B)
    rules={rule_list[i]:p for i,p in enumerate([60,70,65])} # may attracted only if within the dist specified in the value of the rule
    rule_probs={rule_list[i]:p for i,p in enumerate([0.7,0.7,0.8])} # probabilities of attraction if within range

    map=Map(map_width,map_height,types,populations,max_speeds,max_accs,rules,rule_probs)
    print(map.rules)
    n_iters=1000 # originally is 1000
    # suppose we want to study A ->(A,B) in this case
    # a list containing the density for chosen A for all iterations    
    B_density = []
    # a list for (A,B)
    AB_density = []
    ABC_density = []
    ABCD_density=[]
    # DE_density = []
    # DEF_density=[]
    #A_density = []
    for rule in rules.keys():
        type1,type2=rule
        print('%s is attracted by %s'%(type1,type2))
    map.init_GUI()
    for i in tqdm(range(n_iters)):
        map.update_GUI()
        dic1 = map.compute_density(thres = 25, mode=1)
        dic2 = map.compute_density(thres = 25, mode = 2)
        #dic3 = map.compute_density(thres = 20, mode = 1)
        #A_density.append(dic2['A'])
        B_density.append(dic2['B'])
        AB_density.append(dic1[('A','B')])
        ABC_density.append(dic1[('C',('A','B'))])
        ABCD_density.append(dic1[('D',('A','B','C'))])
        # DE_density.append(dic3[('D','E')])
        # DEF_density.append(dic3[('F',('D','E'))])
    tt.done()
    # print(A_density)
    # print("\n")
    print (B_density)
    print("\n")
    print(AB_density)
    print("\n")
    print(ABC_density)
    print("\n")
    print(ABCD_density)
    # print("\n")
    # print(DE_density)
    # print("\n")
    # print(DEF_density)