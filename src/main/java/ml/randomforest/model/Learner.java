package ml.randomforest.model;

/**
 * The learner is responsible for defining the parameter set format and the way
 * to randomly sampling the parameters.
 */
public interface Learner {

    /**
     * @return number of classes / labels for the classification task
     */
    public int nClasses();

    /**
     * returns a parameter set from randomly sampling the parameter space
     * 
     * @return a parameter set, @see package ml.randomforest.model.ParameterSet
     */
    public ParameterSet sample();
}
