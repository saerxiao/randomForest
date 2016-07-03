package ml.randomforest.model;

public class TrainingOptions {

    private final int nTrees;
    private final int nSample;
    private final int treeMaxDepth;
    private final float minInformationGain;
    private final int minDataSizePerNode;

    public TrainingOptions(int nTrees, int nSample, int treeMaxDepth, float minInformationGain,
            int minDataSizePerNode) {
        this.nTrees = nTrees;
        this.nSample = nSample;
        this.treeMaxDepth = treeMaxDepth;
        this.minInformationGain = minInformationGain;
        this.minDataSizePerNode = minDataSizePerNode;
    }

    /**
     *
     * @return the number of trees, default is 5
     */
    public int nTrees() {
        return nTrees;
    }

    /**
     * 
     * @return the number of samples to draw to compute the decision on each
     *         node, default is 100
     */
    public int nSample() {
        return nSample;
    }

    /**
     * 
     * @return maximum depth of the tree, default is 2
     */
    public int treeMaxDepth() {
        return treeMaxDepth;
    }

    /**
     * 
     * @return information gain, equal or smaller than which the tree will stop
     *         growing, default is 0
     */
    public float minInformationGain() {
        return minInformationGain;
    }

    /**
     * 
     * @return minimum data size per node to continue growing the tree
     */
    public int minDataSizePerNode() {
        return minDataSizePerNode;
    }

    public static class Builder {
        private int nTrees = 5;
        private int nSample = 100;
        private int treeMaxDepth = 2;
        private float minInformationGain = 0f;
        private int minDataSizePerNode = 1;

        public Builder nTrees(int nTrees) {
            this.nTrees = nTrees;
            return this;
        }

        public Builder nSample(int nSample) {
            this.nSample = nSample;
            return this;
        }

        public Builder treeMaxDepth(int treeMaxDepth) {
            this.treeMaxDepth = treeMaxDepth;
            return this;
        }

        public Builder minInformationGain(float minInformationGain) {
            this.minInformationGain = minInformationGain;
            return this;
        }

        public Builder minDataSizePerNode(int minDataSizePerNode) {
            this.minDataSizePerNode = minDataSizePerNode;
            return this;
        }

        public TrainingOptions build() {
            return new TrainingOptions(nTrees, nSample, treeMaxDepth, minInformationGain, minDataSizePerNode);
        }
    }
}
