package ml.randomforest.model;

/**
 * Conveniently encapsulates the input (features) and class label
 */
public class DataPoint {

    private int label = -1;
    private float[] features;

    public DataPoint(float[] features) {
        this.features = features;
    }

    public DataPoint(int label, float[] features) {
        this(features);
        this.label = label;
    }

    public int getLabel() {
        return label;
    }

    public float[] getFeatures() {
        return features;
    }
}
