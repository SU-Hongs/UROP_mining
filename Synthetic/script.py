from map import Map
import time
from visualizer import Visualizer
import turtle as tt
from tqdm import tqdm
if __name__ == '__main__':
    width,height=200,200
    map=Map(width=width,height=height,num_1=100,num_2=100)
    visualizer=Visualizer(map=map,types=['A','B'],colors=['red','blue'],shapes=['circle','square'])
    tt.setworldcoordinates(0,0,width,height)
    print('Rule: A (red) attracts B (blue).')
    start_time=time.time()
    for i in tqdm(range(100)):
        visualizer.updateGUI()
        map.move()
    interval=time.time()-start_time
    print('Total seconds to run 1000 iterations:',interval)
    tt.done()