from moving_object import MovingObject

class Map:
    def __init__(self,width,height):
        self.width,self.height=width,height

    ### translate MovingObject by (dx,dy)
    def move(self,obj:MovingObject,dx,dy):
        obj.move(dx,dy)
        obj.setX(obj.getX()%self.width) # concatenate left and right sides
        obj.setY(obj.getY()%self.height) # concatenate upper and lower sides
    
    