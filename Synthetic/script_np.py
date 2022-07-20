import numpy as np
from math import pi, sqrt
import turtle as tt
import matplotlib.cm as cm
from tqdm import tqdm
import csv
import copy
import itertools
import argparse
import time

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
    def __init__(self,map_width,map_height,types,populations,max_speeds,max_accs,rules,rule_probs,colo_thres,unattracted_thres,useGUI=False,eps=1e-7):
        self.map_width,self.map_height=map_width,map_height # width and height of the map
        self.types=types # types of objects
        self.populations=populations # populations of different types
        self.max_speeds=max_speeds # max velocities of different types
        self.max_accs=max_accs # max accelerations of different types
        self.rules=rules # rules of attraction
        self.rule_probs=rule_probs # probabilities of attraction if within range
        self.colo_thres=colo_thres # threshold of distance for colocation
        self.unattracted_thres=unattracted_thres # threshold of distance that stop approaching attracting objects
        self.positions=dict() # dict of arrays of different types of objects
        self.speeds=dict() # dict os arrays of different types of objects
        self.distances=DistDict() # dict of pairwise distances for all rules
        self.useGUI=useGUI # use GUI or not
        self.eps=eps # small number to avoid division by zero
        area_factor=1/2
        len_factor=sqrt(area_factor)
        lb_factor=(1-len_factor)/2
        self.window={'lb_x':int(self.map_width*lb_factor),
            'lb_y':int(self.map_height*lb_factor),
            'length':int(np.minimum(self.map_height,self.map_width)*len_factor)} # window to compute colocation

        self.init_objects()
        self.init_GUI()

    def get_snapshot(self):
        return copy.deepcopy(self.positions),copy.deepcopy(self.speeds),copy.deepcopy(self.distances)
    
    def load_snapshot(self,positions,speeds,distances):
        self.positions,self.speeds,self.distances=positions,speeds,distances
    
    def use_GUI(self):
        return self.useGUI

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
    def flatten_rule(self,rule):
        obj1,objs=rule
        if type(objs)==str: return [obj1,objs]
        elif type(objs)==tuple: return list(objs)+[obj1,]
    
    # Update speed for objects that is attracted by single object
    def single_attraction(self,rule,dist):

        type1,type2=rule
        obj1_pos,obj2_pos=self.positions[type1],self.positions[type2]
        num_type1=self.populations[type1]

        # get random index of type2 that attracts type1, return -1 if not exists 
        attracted_indices=np.apply_along_axis(self.randomIndex,1,(self.distances[rule]<=dist)*(self.distances[rule]>self.unattracted_thres))

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
        attracted_indices=np.concatenate([np.apply_along_axis(self.randomIndex,1,
            (self.distances[(type1,t)]<=dist)*(self.distances[(type1,t)]>self.unattracted_thres)).reshape(-1,1) for t in type2],axis=1)
        
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
        if not self.useGUI: return
        tt.tracer(False)
        tt.setworldcoordinates(0,0,self.map_width,self.map_height*1.1)
        tt.colormode(1.0)
        tt.penup()
        tt.shapesize(0.3,0.3,0.3)
        tt.shape(name='circle')
        self.colors=cm.rainbow(np.linspace(0,1,len(self.types)))[:,:3]
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
            self.textTurtle.fd(0.5*self.map_width/max(len(self.types),5))
        self.textTurtle.ht()
        tt.update()
    
    def update_GUI(self):
        positions=self.iterate()
        if self.useGUI:
            tt.clear()
            for i,t in enumerate(positions.keys()):
                pos=positions[t]
                tt.color(*self.colors[i])
                for p in pos:
                    tt.goto(p[0],p[1])
                    tt.stamp()
            tt.update()

    # thres is the max distance within a colocation
    # idx is a map from type to indices of objects in a window
    # full_list is the list of types for a colocation pattern
    # curr_list is used for recursion
    # colo_list is a 2d array, where each row is a colocation instance of curr_list
    def compute_colocation(self,thres,idx,full_list,curr_list=None,colo_list=None):
        if curr_list==None: curr_list=list()
        if len(full_list)==len(curr_list): return len(colo_list)
        type1 = full_list[len(curr_list)] # new type of object
        objs_t1=idx[type1] # indices of type1 objects in window
        pos_t1=self.positions[type1][objs_t1] # get positions of type1 objects in window

        # base case
        if len(curr_list)==0: # add all objects of type1 within the window
            curr_list.append(type1)
            colo_list=np.array(objs_t1).reshape(-1,1)
            return self.compute_colocation(thres,idx,full_list,curr_list,colo_list)
        
        # general case
        mask=np.ones((len(colo_list),len(objs_t1))) # mask[i,j] checks whether colo_list[i] and objs_t1[j] form a colocation
        for i,type2 in enumerate(curr_list):
            objs_t2=colo_list[:,i]
            pos_t2=self.positions[type2][objs_t2] # get positions of type2 objects in window
            # pairwise distances of shape = (# of type2) * (# of type1)
            dists=np.sqrt(-2*pos_t2.dot(pos_t1.T)+np.sum(pos_t2**2,axis=1,keepdims=True)+np.sum(pos_t1**2,axis=1))
            mask*=(dists<=thres) # check whether type1 and type2 has distance less than thres
        curr_list.append(type1)
        colo_list=np.array(
            [list(colo_list[idx_colo])+[objs_t1[idx_t1]] 
            for idx_colo,row in enumerate(mask) 
                for idx_t1,val in enumerate(row) 
                    if val==True]) # generate new colo_list for curr_list
        if len(colo_list)==0: return 0 # if empty, return 0
        return self.compute_colocation(thres,idx,full_list,curr_list,colo_list)

    # return a map from (T1,T2,...) to number of colocations
    def compute_colocations_mpi(self,thres,idx):
        colocations={} # map from colocation pattern to num of colocation
        mpis={} # map from colocation pattern to modified participation index
        colo_lists={} # map from colcoation pattern to ndarray of colocation instances
        
        # base case
        for t in self.types:
            colocations[(t,)]=len(idx[t])
            mpis[(t,)]=int(len(idx[t])!=0) # 1 if there exist objects of type t in window, else 0
            colo_lists[(t,)]=np.array(idx[t]).reshape(-1,1)

        # general case
        for L in range(2,len(self.types)+1):
            for subset in itertools.combinations(self.types,L):
                full_list=sorted(list(subset))
                curr_list=full_list[:-1]
                if tuple(curr_list) not in colo_lists: # no colocation for curr_list
                    colocations[tuple(full_list)]=0
                    mpis[tuple(full_list)]=0 # 0 if no colocation in window
                    continue
                colo_list=colo_lists[tuple(curr_list)] # there exists colocation for curr_list

                type1 = full_list[len(curr_list)] # new type of object
                objs_t1=idx[type1] # indices of type1 objects in window
                pos_t1=self.positions[type1][objs_t1] # get positions of type1 objects in window

                mask=np.ones((len(colo_list),len(objs_t1))) # mask[i,j] checks whether colo_list[i] and objs_t1[j] form a colocation
                for i,type2 in enumerate(curr_list):
                    objs_t2=colo_list[:,i]
                    pos_t2=self.positions[type2][objs_t2] # get positions of type2 objects in window
                    # pairwise distances of shape = (# of type2) * (# of type1)
                    dists=np.sqrt(-2*pos_t2.dot(pos_t1.T)+np.sum(pos_t2**2,axis=1,keepdims=True)+np.sum(pos_t1**2,axis=1))
                    mask*=(dists<=thres) # check whether type1 and type2 has distance less than thres
                colo_list=np.array(
                    [list(colo_list[idx_colo])+[objs_t1[idx_t1]] 
                    for idx_colo,row in enumerate(mask) 
                        for idx_t1,val in enumerate(row) 
                            if val==True]) # generate new colo_list for curr_list
                if len(colo_list)==0: 
                    colocations[tuple(full_list)]=0
                    mpis[tuple(full_list)]=0 # 0 if no colocation in window
                else:
                    colocations[tuple(full_list)]=len(colo_list)
                    mpis[tuple(full_list)]=max([len(set(colo_list[:,i]))/len(idx[t]) for i,t in enumerate(full_list)]) # modified PI
                    if len(full_list)>2: # calculate the participation ratios of all proper subsets
                        for i in range(2,len(full_list)):
                            for t in itertools.combinations(list(range(len(full_list))),i):
                                mpis[tuple(full_list)]=max(mpis[tuple(full_list)],
                                    len(set(map(tuple,colo_list[:,t])))/colocations[tuple(np.array(full_list)[list(t)])])
                                    
                    colo_lists[tuple(full_list)]=colo_list
        return colocations,mpis

    def compute_density(self,thres,full_list):
        lb_x,lb_y,length=self.window['lb_x'],self.window['lb_y'],self.window['length']
        idx_dict = dict()
        for t in self.types:
            x, y = self.positions[t][:,0], self.positions[t][:,1]
            idx = np.where((x>=lb_x) & (x<lb_x+length) & (y>=lb_y) & (y<lb_y+length))[0]
            idx_dict[t] = idx
        return self.compute_colocation(thres,idx_dict,full_list)

    # a function to select the objects in the targeted sub-region
    # return a dictionary of the the colocation density of the colocation patterns 
    def compute_densities_mpi(self, thres):
        # first select the index of the objects that is in the range of the targeted subregion
        # define the left bottom corner of the sub-region and the length
        lb_x,lb_y,length=self.window['lb_x'],self.window['lb_y'],self.window['length']
        idx_dict = dict()
        for t in self.types:
            x, y = self.positions[t][:,0], self.positions[t][:,1]
            idx = np.where((x>=lb_x) & (x<lb_x+length) & (y>=lb_y) & (y<lb_y+length))[0]
            idx_dict[t] = idx
        return self.compute_colocations_mpi(thres,idx_dict)

    # compute modified participation index
    def compute_mpi(self):

        pass

def write_to_csv(fname,dic):
    with open(fname,'w',newline='') as csvfile:
        fieldnames = [k for k in dic.keys()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        length=len(dic[fieldnames[0]])
        for i in range(length):
            writer.writerow({name:dic[name][i] for name in fieldnames})

def generate_data(colo_path,mpi_path,useGUI):
    # Initialization of map
    map_width,map_height=2000,2000 # width and height of the map
    types=['A','B','C','D'] # types of objects
    mean,sd=100,20
    p_nums=list(np.round(np.maximum(np.random.randn(len(types))*sd+mean,0)).astype(int))
    print(p_nums)
    populations={types[i]:v for i,v in enumerate(p_nums)} # populations of different types
    max_speeds={types[i]:v for i,v in enumerate([5,5,5,5])} # max velocities of different types
    max_accs={types[i]:v for i,v in enumerate([0.5,0.5,0.5,0.5])} # max accelerations of different types
    rule_list=[('A','B'),('C',('A','B'))] # list of rules where the first is attracted by the second (e.g. (A,B) means A->B)
    rules={rule_list[i]:p for i,p in enumerate([50,50])} # may attracted only if within the dist specified in the value of the rule
    rule_probs={rule_list[i]:p for i,p in enumerate([1,1])} # probabilities of attraction if within range
    unattracted_thres=10 # threshold of distance that stop approaching attracting objects
    colo_thres=25 # threshold of distance for colocation
    time_granularity=1 # freqency of computing num of colocations

    map=Map(map_width,map_height,types,populations,max_speeds,max_accs,rules,rule_probs,colo_thres,unattracted_thres,useGUI)
    print(map.rules)
    start_calc=1000 # num of iters to stabilize and start calculation of colo and mPI 
    n_iters=1000 # num of iters after start_calc
    # suppose we want to study A ->(A,B) in this case
    # a list containing the density for chosen A for all iterations    
    densities,mpis={},{}
    for rule in rules.keys():
        type1,type2=rule
        print('%s is attracted by %s'%(type1,type2))
    map.init_GUI()
    for i in tqdm(range(start_calc+n_iters)):
        map.update_GUI()
        if i%time_granularity==0 and i>=start_calc:
            colo_nums,mparts = map.compute_densities_mpi(thres = colo_thres)
            if(len(densities.keys())==0): 
                densities.update({''.join(k):[] for k in colo_nums})
                mpis.update({''.join(k):[] for k in mparts})
            for key,val in colo_nums.items(): densities[''.join(key)].append(val)
            for key,val in mparts.items(): mpis[''.join(key)].append(val)
    if map.use_GUI(): tt.done()
    write_to_csv(colo_path,densities)
    write_to_csv(mpi_path,mpis)

if __name__=='__main__':

    # run "python script_np.py --gui" to enable GUI
    # run "python script_np.py" to disable GUI 

    parser = argparse.ArgumentParser(description='Simulator for Moving Objects')
    parser.add_argument('--gui', action='store_true')
    parser.set_defaults(gui=False)
    args = parser.parse_args()

    n_times=100
    
    for i in range(1,n_times+1):
        generate_data('data/simu_colo%s.csv'%str(i).zfill(len(str(n_times))),
            'data/simu_mpi%s.csv'%str(i).zfill(len(str(n_times))),useGUI=args.gui)
