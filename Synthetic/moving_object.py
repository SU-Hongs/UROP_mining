class MovingObject:
    def __init__(self,x,y,type):
        self.x,self.y,self.type=x,y,type
    def move(self,dx,dy):
        self.x+=dx
        self.y+=dy
    def setX(self,x): self.x=x
    def setY(self,y): self.y=y
    def getX(self): return self.x
    def getY(self): return self.y