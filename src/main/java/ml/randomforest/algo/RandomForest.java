package ml.randomforest.algo;

import ml.randomforest.model.DataPoint;
import ml.randomforest.model.DecisionTreeNode;
import ml.randomforest.model.Learner;
import ml.randomforest.model.TrainingOptions;

public class RandomForest {

    /**
     * 
     * @param data
     *            an array of training data
     * @param learner
     *            the learner definition, @see ml.randomforest.model.Learner
     * @param trainingOption
     *            @see ml.randomforest.model.TrainingOption
     * @return an array of the decision tree root nodes
     */
    public static DecisionTreeNode[] trainClassifier(DataPoint[] data, Learner learner,
            TrainingOptions trainingOption) {
        DecisionTreeNode[] forest = new DecisionTreeNode[trainingOption.nTrees()];
        for (int i = 0; i < forest.length; i++) {
            forest[i] = RandomDecisionTree.trainClassifier(data, learner, trainingOption);
        }
        return forest;
    }

    /**
     * 
     * @param dataPoint
     *            the data to test
     * @param forest
     *            an array of the decision tree root nodes
     * @return the distribution mass function
     */
    public static float[] classify(DataPoint dataPoint, DecisionTreeNode[] forest) {
        int nClasses = forest[0].getHistogram().length;
        float[] p = new float[nClasses];
        for (int t = 0; t < forest.length; t++) {
            float[] pt = RandomDecisionTree.classify(dataPoint, forest[t]);
            for (int c = 0; c < p.length; c++) {
                p[c] += pt[c];
            }
        }
        for (int c = 0; c < p.length; c++) {
            p[c] = p[c] / forest.length;
        }

        return p;
    }
}
