package ml.randomforest.algo;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestRandomDecisionTree {

    @Test
    public void testEntropy() {
        int N = 10;
        int[] hist1 = { 5, 5 };
        assertEquals(0.5f * -Math.log(0.5f) * 2, RandomDecisionTree.entropy(N, hist1), 1e-5f);

        int[] hist2 = { 10, 0 };
        assertEquals(0, RandomDecisionTree.entropy(N, hist2), 0);
    }

    @Test
    public void testEntropyForEmptySet() {
        int N = 0;
        int[] hist = { 0, 0 };
        assertEquals(0f, RandomDecisionTree.entropy(N, hist), 0f);
    }

    @Test
    public void testInformationGain() {
        int N = 10;
        int[] hist = { 5, 5 };
        float currentE = RandomDecisionTree.entropy(N, hist);
        int left = 5;
        int[] leftHist = { 5, 0 };
        int right = 5;
        int[] rightHist = { 0, 5 };
        assertEquals(currentE, RandomDecisionTree.informationGain(currentE, leftHist, left, rightHist, right), 1e-5f);

        left = 10;
        leftHist = new int[] { 5, 5 };
        right = 0;
        rightHist = new int[] { 0, 0 };
        assertEquals(0, RandomDecisionTree.informationGain(currentE, leftHist, left, rightHist, right), 0);
    }
}
