package ml.randomforest.model;

public class DecisionTreeNode {
    private ParameterSet param;

    private DecisionTreeNode left;

    private DecisionTreeNode right;

    private int nTotal;

    private final int[] histogram;

    public DecisionTreeNode(int nClasses) {
        histogram = new int[nClasses];
    }

    public ParameterSet getParam() {
        return param;
    }

    public void setParam(ParameterSet param) {
        this.param = param;
    }

    public DecisionTreeNode getLeft() {
        return left;
    }

    public void setLeft(DecisionTreeNode left) {
        this.left = left;
    }

    public DecisionTreeNode getRight() {
        return right;
    }

    public void setRight(DecisionTreeNode right) {
        this.right = right;
    }

    public int getNTotal() {
        return nTotal;
    }

    public void setNTotal(int nTotal) {
        this.nTotal = nTotal;
    }

    public int[] getHistogram() {
        return histogram;
    }

    public float[] getDistribution() {
        if (nTotal > 0) {
            float[] dist = new float[histogram.length];
            for (int i = 0; i < histogram.length; i++) {
                dist[i] = histogram[i] * 1.0f / nTotal;
            }
            return dist;
        } else {
            return null;
        }
    }

    @Override
    public String toString() {
        return toString(this);
    }

    private String toString(DecisionTreeNode node) {
        StringBuilder str = new StringBuilder();
        if (node.getParam() != null) {
            str.append(node.getParam().toString()).append("\n");
            str.append(toString(node.getLeft()));
            str.append(toString(node.getRight()));
        } else {
            for (int c = 0; c < node.getHistogram().length; c++) {
                str.append(node.getHistogram()[c]).append(" ");
            }
            str.append("\n");
        }
        return str.toString();
    }
}
