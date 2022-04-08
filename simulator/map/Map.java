package simulator.map;

import simulator.objects.GeneralObject;

public class Map {
    
    private float width,height;
    private String[] types;
    private GeneralObject[] objects;
    
    public Map(float width, float height, String[] types){
        this.width=width;
        this.height=height;
        this.types=types;
    }

    public float getWidth(){return this.width;}
    public float getHeight(){return this.height;}
    public String[] getTypes(){return this.types;}
    public GeneralObject[] getObjects(){return this.objects;}
}
