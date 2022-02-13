import turtle as tt
from map import Map
class Visualizer():
    def __init__(self,map:Map,types=['A','B'],colors=['red','blue'],shapes=['circle','square']):
        self.map=map
        self.types=types
        self.colors=colors
        self.shapes=shapes
    def updateGUI(self):
        objsList=self.map.getObjects()
        tt.clearscreen()
        tt.tracer(False)
        for i,objs in enumerate(objsList):
            tt.shape(name=self.shapes[i])
            tt.shapesize(0.3,0.3,0.3)
            tt.color(self.colors[i])
            for obj in objs:
                tt.penup()
                tt.setpos(obj.getX(),obj.getY())
                tt.pendown()
                tt.stamp()
        
