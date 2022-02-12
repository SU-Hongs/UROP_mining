from moving_object import *
import numpy as np
import random

# here one map only has two types of objects
# the size of map should be at least 10x10
class Map:
    # a map has two list of obejcts
    def __init__(self,width=80,height=80,num_1=100,num_2=100):
        self.width,self.height=width,height
        self.obj1 = []
        self.obj2 = []
        # x corresponds to width, y corresponds to height
        for i in range(num_1):
            x = random.randint(0,width)
            y = random.randint(0,height)
            obj = MovingObject(x,y,"A")
            self.obj1.append(obj)
        for i in range(num_2):
            x = random.randint(0,width)
            y = random.randint(0,height)
            obj = MovingObject(x,y,"B")
            self.obj2.append(obj)


    # a helper function to check whether an object occupies an valid position
    def check_position(self,x,y):
        if x<0 or y<0 or x>= self.width or y >= self.height:
            return False
        else:
            return True

    # a helper function to check whether certain position is occupied by any objects of certain type 
    def check_occupy(self,x,y,objs):
        if not self.check_position(x,y):
            return False
        for obj in objs:
            obj_x = obj.getX()
            obj_y = obj.getY()
            if x == obj_x and y == obj_y:
                return True
        return False

    ### this move function will update all the objects on the map once by each call
    def move(self):
        # the moving stage consists of two parts
        # suppose type "A" objects will attract type "B" objects
        # Part 1: A will randomly walk around the map
        #         B will randomly walk around the map unless there exists A objects nearby (7x7)

        # first move objects A
        for obj in self.obj1:

            obj_x = obj.getX()
            obj_y = obj.getY()
            # dx, dy are -1, 0 or 1
            dx = random.randint(-1,2)
            dy = random.randint(-1,2)
            obj_x += dx
            obj_y += dy
            # if the moving position is valid, then move
            # otherwise the object will keep still
            if self.check_position(obj_x,obj_y):
                obj.setX(obj_x)
                obj.setY(obj_y)

        # then move objects B
        for objj in self.obj2:
            # objects B will move towards the first object A in its neighbourhood (3x3)
            attract = False
            x = objj.getX()
            y = objj.getY()
            for dx in range(-3,4):
                for dy in range(-3,4):
                    if (attract):
                        continue
                    x_new = x+dx
                    y_new = y+dy
                    if self.check_occupy(x_new,y_new,self.obj1):
                        # current object B has 60% chance to be attracted by A
                        if random.random()<0.6:
                            attract=True
                            x+=np.sign(dx)
                            y+=np.sign(dy)
                            if self.check_position(x,y):
                                objj.setX(x)
                                objj.setY(y)
            # not attract implies there is no A in the neighbourhood of current B
            # then this object B just random walk
            if not attract:
                # dx, dy are -1, 0 or 1
                dx = random.randint(-1,2)
                dy = random.randint(-1,2)
                x += dx
                y += dy
                # if the moving position is valid, then move
                # otherwise the object will keep still
                if self.check_position(x,y):
                    obj.setX(x)
                    obj.setY(y)


                    
        
        # obj.move(dx,dy)
        # obj.setX(obj.getX()%self.width) # concatenate left and right sides
        # obj.setY(obj.getY()%self.height) # concatenate upper and lower sides

    # a helper function to check whether the current object is in the sub-region
    def in_subregion(self,corner_x, corner_y, length,x,y):
        if x>=corner_x and x<=(corner_x+length):
            if y>=corner_y and y<= (corner_y+length):
                return True
        return False


        # return the objects in the sub-regions
    def select_objs(self):
        # left-top position of the selected sub-region
        corner_x = int(self.width/4)
        corner_y = int(self.height/4)
        length = int(np.minimum(self.width,self.height)/3)
        select_obj1 = []
        select_obj2 = []
        for obj in self.obj1:
            if self.in_subregion(corner_x,corner_y,length,obj.getX(),obj.getY()):
                select_obj1.append(obj)
        for obj in self.obj2:
            if self.in_subregion(corner_x,corner_y,length,obj.getX(),obj.getY()):
                select_obj2.append(obj)
        return select_obj1,select_obj2



        


    
    
    