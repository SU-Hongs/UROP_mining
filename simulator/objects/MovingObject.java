package simulator.objects;

public class MovingObject extends GeneralObject {
    
    // Constructor
    public MovingObject(String type,float x,float y,boolean isAlive){
        super(type,x,y,isAlive);
    }

    // Setter
    public void move(float dx,float dy){
        this.x+=dx;
        this.y+=dy;
    }
}
