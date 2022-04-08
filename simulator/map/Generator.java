package simulator.map;

import java.util.Random;

/** Random generator, a utility class */
public class Generator {
    
    private Random r;
    private long seed;

    public Generator(long seed){
        r = new Random(seed);
    }
    public Generator(){
        this.seed = System.currentTimeMillis();
        r = new Random(this.seed);
    }

    /** Sampling from N(0,1) */
    public double nextGaussian(){ return r.nextGaussian(); }
    
    /** Sampling from N(mean,std^2) */
    public double nextGaussian(double mean,double std){ return mean + std * r.nextGaussian(); }
    
    /** Sampling from Unif(0,1) */
    public double nextUniform(){ return r.nextDouble(); }

    /** Sampling from Unif(a,b) */
    public double nextUniform(double a,double b){ return r.nextDouble() * (b - a) + a; }

}
