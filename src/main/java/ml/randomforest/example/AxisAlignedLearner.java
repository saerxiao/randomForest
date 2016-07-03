package ml.randomforest.example;

import java.util.Random;

import ml.randomforest.model.DataPoint;
import ml.randomforest.model.ParameterSet;
import ml.randomforest.model.Learner;

/**
 * Learner where each decision node is a threshold for a single feature (i.e.
 * along a single axis)
 */
public class AxisAlignedLearner implements Learner {

    private final int nClasses;
    private final float[][] rangeToSampleThreshold;
    private final Random random = new Random(System.currentTimeMillis());

    public AxisAlignedLearner(int nClasses, float[][] rangeToSampleThreshold) {
        this.nClasses = nClasses;
        this.rangeToSampleThreshold = rangeToSampleThreshold;
    }

    @Override
    public int nClasses() {
        return nClasses;
    }

    @Override
    public ParameterSet sample() {
        int axis = random.nextInt(rangeToSampleThreshold.length);
        return new AxisAlignedParameterSet(axis, rangeToSampleThreshold[axis][0]
                + random.nextFloat() * (rangeToSampleThreshold[axis][1] - rangeToSampleThreshold[axis][0]));
    }

    public static class AxisAlignedParameterSet implements ParameterSet {
        private final int axis;
        private final float threshold;

        public AxisAlignedParameterSet(int axis, float threshold) {
            this.axis = axis;
            this.threshold = threshold;
        }

        @Override
        public boolean evaluateDecision(DataPoint dataPoint) {
            return dataPoint.getFeatures()[axis] < threshold;
        }

        public int getAxis() {
            return axis;
        }

        public float getThreshold() {
            return threshold;
        }

        @Override
        public String toString() {
            return axis + " " + threshold;
        }
    }
}
