package ml.randomforest.algo;

import java.util.Optional;
import java.util.stream.IntStream;

import ml.randomforest.model.DataPoint;
import ml.randomforest.model.DecisionTreeNode;
import ml.randomforest.model.ParameterSet;
import ml.randomforest.model.Learner;
import ml.randomforest.model.TrainingOptions;

public class RandomDecisionTree {

    /**
     * 
     * @param data
     *            an array of training data
     * @param learner
     *            the learner definition, @see ml.randomforest.model.Learner
     * @param trainingOptions
     * @see ml.randomforest.model.TrainingOptions
     * @return the decision tree root node
     */
    public static DecisionTreeNode trainClassifier(DataPoint[] data, Learner learner, TrainingOptions trainingOptions) {
        return trainClassifier(data, learner, trainingOptions, 1, 0, data.length);
    }

    /**
     * 
     * @param dataPoint
     *            the data point to test
     * @param node
     *            the root node of the decision tree
     * @return the probability mass function
     */
    public static float[] classify(DataPoint dataPoint, DecisionTreeNode node) {
        // leaf node
        if (node.getParam() == null)
            return node.getDistribution();

        if (node.getParam().evaluateDecision(dataPoint)) {
            return classify(dataPoint, node.getLeft());
        } else {
            return classify(dataPoint, node.getRight());
        }
    }

    static DecisionTreeNode trainClassifier(DataPoint[] data, Learner learner, TrainingOptions trainingOption,
            int depth, int lowInclusive, int hiExclusive) {
        DecisionTreeNode node = new DecisionTreeNode(learner.nClasses());
        node.setNTotal(hiExclusive - lowInclusive);
        for (int i = lowInclusive; i < hiExclusive; i++) {
            node.getHistogram()[data[i].getLabel()]++;
        }
        if (depth == trainingOption.treeMaxDepth() || hiExclusive - lowInclusive < trainingOption.minDataSizePerNode())
            return node;

        float nodeEntropy = entropy(node.getNTotal(), node.getHistogram());
        final ParameterSet[] params = new ParameterSet[trainingOption.nSample()];
        final float[] infoGains = new float[trainingOption.nSample()];
        // This parallel-for is typically slower for smaller data set because of
        // multi-threaded overhead. For larger data set, it's marginally faster
        // because the performance is memory access bound instead of CPU bound
        // for simple decision nodes
        IntStream.range(0, trainingOption.nSample()).parallel().forEach(k -> {
            ParameterSet param = learner.sample();
            int[] leftHist = new int[learner.nClasses()];
            int[] rightHist = new int[learner.nClasses()];
            int left = 0, right = 0;
            for (int i = lowInclusive; i < hiExclusive; i++) {
                if (param.evaluateDecision(data[i])) {
                    leftHist[data[i].getLabel()]++;
                    left++;
                } else {
                    rightHist[data[i].getLabel()]++;
                    right++;
                }
            }
            params[k] = param;
            infoGains[k] = informationGain(nodeEntropy, leftHist, left, rightHist, right);
        });
        float maxInfoGain = 0f;
        Optional<ParameterSet> bestParameterSet = Optional.empty();
        for (int k = 0; k < infoGains.length; k++) {
            if (infoGains[k] > maxInfoGain) {
                maxInfoGain = infoGains[k];
                bestParameterSet = Optional.of(params[k]);
            }
        }

        if (maxInfoGain <= trainingOption.minInformationGain())
            return node;

        int partitionIndex = partition(data, lowInclusive, hiExclusive,
                bestParameterSet.orElseThrow(IllegalStateException::new));
        node.setParam(bestParameterSet.get());
        node.setLeft(trainClassifier(data, learner, trainingOption, depth + 1, lowInclusive, partitionIndex));
        node.setRight(trainClassifier(data, learner, trainingOption, depth + 1, partitionIndex, hiExclusive));
        return node;
    }

    static float entropy(int N, int[] hist) {
        if (N == 0)
            return 0f;

        float e = 0f;
        for (int i = 0; i < hist.length; i++) {
            if (hist[i] > 0) {
                float p = hist[i] * 1.0f / N;
                e -= p * Math.log(p);
            }
        }
        return e;
    }

    static float informationGain(float nodeEntropy, int[] leftHist, int left, int[] rightHist, int right) {
        float leftEntropy = entropy(left, leftHist);
        float rightEntropy = entropy(right, rightHist);
        float leftWeight = left * 1.0f / (left + right);
        return nodeEntropy - (leftWeight * leftEntropy + (1 - leftWeight) * rightEntropy);
    }

    static int partition(DataPoint[] data, int lowInclusive, int hiExclusive, ParameterSet param) {
        int p = lowInclusive;
        for (int i = lowInclusive; i < hiExclusive; i++) {
            if (param.evaluateDecision(data[i])) {
                DataPoint tmp = data[p];
                data[p] = data[i];
                data[i] = tmp;
                p++;
            }
        }
        return p;
    }
}
