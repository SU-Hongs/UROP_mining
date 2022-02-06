class MovingObject:
    def __init__(self,x,y,type,map):
        self.x,self.y,self.type,self.map=x,y,type,map
    def move(self,dx,dy,map_width,map_height):
        self.x=(self.x+dx)%map_width
        self.y+=dy