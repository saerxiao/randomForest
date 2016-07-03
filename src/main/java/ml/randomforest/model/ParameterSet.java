package ml.randomforest.model;

public interface ParameterSet {

    /**
     * returns the decision of which branch the data should go to on a decision
     * tree node
     * 
     * @param dataPoint
     *            the data to evaluate
     * @return a boolean to indicate to go to the left or right child node
     */
    public boolean evaluateDecision(DataPoint dataPoint);
}
