package simulator.map;

/** A class to determine the strategy of object behaviors, according to rules */
public class Strategy {
    private Rule[] rules;
    public Strategy(Rule[] rules){
        this.rules=rules;
    }
}
