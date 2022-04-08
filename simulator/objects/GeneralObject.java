package simulator.objects;

public class GeneralObject {
    
    protected String type;
    protected float x,y;
    protected boolean isAlive;
    
    // Constructor
    public GeneralObject(String type,float x,float y,boolean isAlive){
        this.type=type;
        this.x=x;
        this.y=y;
        this.isAlive=isAlive;
    }

    // Getter
    public String getType() {return this.type;}
    public float[] getLocation(){ return new float[]{this.x,this.y};}
    public float getX(){return this.x;}
    public float getY(){return this.y;}
    public boolean isAlive(){return this.isAlive;}

}
