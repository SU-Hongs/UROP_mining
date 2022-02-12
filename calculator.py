from map import *
import numpy as np

class Calculator:
    def __init__(self):
        pass

    # calculating the modified PI
    # the default distance threshold for determining colocation pattern 3x3
    def compute_PI(self,map):
        obj1,obj2 = map.select_objs()
        count_A = 0
        count_B = 0
        # counting the number of objects A that are in a colocation
        for obj in obj1:
            for dx in range(-1,2):
                for dy in range(-1,2):
                    x_new = obj.getX()+dx
                    y_new = obj.getY()+dy
                    if map.check_occupy(x_new,y_new,obj2):
                        count_A+=1

        # counting the number of objects B that are in a colocation
        for obj in obj2:
            for dx in range(-1,2):
                for dy in range(-1,2):
                    x_new = obj.getX()+dx
                    y_new = obj.getY()+dy
                    if map.check_occupy(x_new,y_new,obj1):
                        count_B+=1
        Pr_A = count_A/len(obj1)
        Pr_B = count_B/len(obj2)
        PI = np.maximum(Pr_A,Pr_B)
        return Pr_A, Pr_B, PI

    
