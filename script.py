from map import Map
import time
from visualizer import Visualizer
import turtle as tt
if __name__ == '__main__':
    width,height=200,200
    map=Map(width=width,height=height,num_1=100,num_2=100)
    visualizer=Visualizer(map=map,types=['A','B'])
    tt.setworldcoordinates(0,0,width,height)
    for i in range(100):
        visualizer.updateGUI()
        tt.update()
        map.move()
    tt.done()