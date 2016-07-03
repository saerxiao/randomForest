package ml.randomforest;

import static org.junit.Assert.*;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;

import org.junit.Test;

import ml.randomforest.algo.RandomDecisionTree;
import ml.randomforest.algo.RandomForest;
import ml.randomforest.example.AxisAlignedLearner;
import ml.randomforest.example.AxisAlignedLearner.AxisAlignedParameterSet;
import ml.randomforest.model.DataPoint;
import ml.randomforest.model.DecisionTreeNode;
import ml.randomforest.model.Learner;
import ml.randomforest.model.TrainingOptions;

public class TestAxisAlignedLearner {

    private static final Random random = new Random(System.currentTimeMillis());

    @Test
    public void testDecisionTreeUinformSample2Classes() {
        int nClasses = 2;
        float[][][] sampleRangesForAllClasses = { { { 1, 2 }, { 1, 2 } }, { { 4, 5 }, { 1, 2 } } };
        UniformSampleGenerator uniformSampleGenerator = new UniformSampleGenerator(sampleRangesForAllClasses);
        int nTraining = 500;
        DataPoint[] trainingSet = generateData(nTraining, uniformSampleGenerator);
        float[][] rangeToSample = { { 0, 6 }, { 0, 3 } };
        Learner learner = new AxisAlignedLearner(nClasses, rangeToSample);
        TrainingOptions trainingOption = new TrainingOptions.Builder().build();
        DecisionTreeNode tree = RandomDecisionTree.trainClassifier(trainingSet, learner, trainingOption);

        AxisAlignedParameterSet param = (AxisAlignedParameterSet) tree.getParam();
        assertTrue(param.getAxis() == 0);

        int nTest = 100;
        DataPoint[] testSet = generateData(nTest, uniformSampleGenerator);
        int nHits = 0;
        for (int i = 0; i < testSet.length; i++) {
            int c = mostLikelyClass(RandomDecisionTree.classify(testSet[i], tree));
            if (c == testSet[i].getLabel())
                nHits++;
        }
        float accuracy = nHits * 1.0f / nTest * 100;
        assertEquals(100, accuracy, 1e-5f);
        System.out.println(String.format("testDecisionTreeUinformSample2Classes accuracy: %.0f%%", accuracy));
    }

    @Test
    public void testRandomForestUniformSample() {
        int nClasses = 4;
        int nTraining = 1000;
        int nTest = 100;
        float expectedAccuracy = 100;
        float[][] rangeToSample = { { 0, 6 }, { 0, 9 } };
        Learner learner = new AxisAlignedLearner(nClasses, rangeToSample);
        TrainingOptions trainingOption = new TrainingOptions.Builder().nTrees(4).treeMaxDepth(5).build();
        float[][][] sampleRangesForAllClasses = { { { 1, 2 }, { 1, 2 } }, { { 4, 5 }, { 1, 3 } },
                { { 1, 2 }, { 3, 5 } }, { { 4, 5 }, { 5, 8 } } };
        SampleGenerator sampleGenerator = new UniformSampleGenerator(sampleRangesForAllClasses);
        DataPoint[] trainingSet = generateData(nTraining, sampleGenerator);
        DataPoint[] testSet = generateData(nTest, sampleGenerator);
        DecisionTreeNode[] forest = testRandomForest(trainingSet, testSet, learner, trainingOption, expectedAccuracy);
        saveVisualizationImage("uniform4ClassesAxisAlignedLearner.png", trainingSet, rangeToSample, forest);
    }

    @Test
    public void testSpiralSample() {
        int nClasses = 5;
        int nTraining = 1000;
        int nTest = 100;
        float[][] rangeToSample = { { 0, 40 }, { 0, 40 } };
        float expectedAccuracy = 90;
        Learner learner = new AxisAlignedLearner(nClasses, rangeToSample);
        TrainingOptions trainingOption = new TrainingOptions.Builder().nSample(10000).nTrees(10).treeMaxDepth(10)
                .build();
        SampleGenerator sampleGenerator = new Spiral2DSampleGenerator(nClasses);
        DataPoint[] trainingSet = generateData(nTraining, sampleGenerator);
        DataPoint[] testSet = generateData(nTest, sampleGenerator);
        DecisionTreeNode[] forest = testRandomForest(trainingSet, testSet, learner, trainingOption, expectedAccuracy);
        saveVisualizationImage("spiral5ClassesAxisAlignedLearner.png", trainingSet, rangeToSample, forest);

    }

    private static DecisionTreeNode[] testRandomForest(DataPoint[] trainingSet, DataPoint[] testSet, Learner learner,
            TrainingOptions trainingOption, float expectedAccuracy) {
        long before = System.currentTimeMillis();
        DecisionTreeNode[] forest = RandomForest.trainClassifier(trainingSet, learner, trainingOption);
        System.out.println(String.format("testRandomForest training took %d ms", System.currentTimeMillis() - before));

        int nHits = 0;
        for (int i = 0; i < testSet.length; i++) {
            int c = mostLikelyClass(RandomForest.classify(testSet[i], forest));
            if (c == testSet[i].getLabel())
                nHits++;
        }
        float accuracy = nHits * 1.0f / testSet.length * 100;
        assertTrue(accuracy >= expectedAccuracy);
        System.out.println(String.format("testRandomForest test accuracy: %.0f%%", accuracy));
        return forest;
    }

    // Draw the classifier output (in color) along a uniform grid across the
    // data range, and plot the data points on top.
    private static void saveVisualizationImage(String fileName, DataPoint[] data, float[][] dataRange,
            DecisionTreeNode[] forest) {
        Component c = new GraphPanel(data, dataRange, forest);
        JFrame frame = new JFrame();
        frame.setBackground(Color.WHITE);
        frame.setUndecorated(true);
        frame.getContentPane().add(c);
        frame.pack();

        BufferedImage bi = new BufferedImage(c.getWidth(), c.getHeight(), BufferedImage.TYPE_INT_ARGB);
        Graphics2D graphics = bi.createGraphics();
        c.print(graphics);
        graphics.dispose();

        try {
            File outputfile = new File(fileName);
            ImageIO.write(bi, "png", outputfile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static class GraphPanel extends JPanel {
        /**
         * eclipsed generated
         */
        private static final long serialVersionUID = -2811304530003735829L;
        private static final int BORDER_GAP = 30;
        private static final int nGridX = 50;
        private static final int nGridY = 50;
        private static final int PREF_W = 800;
        private static final int PREF_H = 650;
        private static final String[] MARKERS = { "o", "X", "*", "#", "Z" };
        private static final Color[] COLORS = { Color.RED, Color.GREEN, Color.YELLOW, Color.BLUE, Color.MAGENTA };

        private final DataPoint[] data;
        private final float[][] dataRange;
        private final DecisionTreeNode[] forest;

        public GraphPanel(DataPoint[] data, float[][] dataRange, DecisionTreeNode[] forest) {
            this.data = data;
            this.dataRange = dataRange;
            this.forest = forest;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            drawDistribution(g2);
            drawData(g2);
        }

        @Override
        public Dimension getPreferredSize() {
            return new Dimension(PREF_W, PREF_H);
        }

        private void drawDistribution(Graphics2D g2) {
            double xScale = ((double) getWidth() - 2 * BORDER_GAP) / nGridX;
            double yScale = ((double) getHeight() - 2 * BORDER_GAP) / nGridY;

            for (int i = 0; i < nGridX; i++) {
                for (int j = 0; j < nGridY; j++) {
                    float x = (i + 0.5f) / nGridX * (dataRange[0][1] - dataRange[0][0]);
                    float y = (j + 0.5f) / nGridY * (dataRange[1][1] - dataRange[1][0]);
                    DataPoint dataPoint = new DataPoint(new float[] { x, y });
                    float[] classesDistribution = RandomForest.classify(dataPoint, forest);
                    float red = 0, green = 0, blue = 0;
                    for (int c = 0; c < classesDistribution.length; c++) {
                        red += COLORS[c].getRed() * classesDistribution[c];
                        green += COLORS[c].getGreen() * classesDistribution[c];
                        blue += COLORS[c].getBlue() * classesDistribution[c];
                    }
                    red = Math.max(0, Math.min(red / 255, 1));
                    green = Math.max(0, Math.min(green / 255, 1));
                    blue = Math.max(0, Math.min(blue / 255, 1));
                    Color color = new Color(red, green, blue);
                    g2.setColor(color);
                    g2.fillRect((int) (i * xScale) + BORDER_GAP, (int) (j * yScale) + BORDER_GAP,
                            (int) Math.ceil(xScale), (int) Math.ceil(yScale));

                }
            }
        }

        private void drawData(Graphics2D g2) {
            g2.setColor(Color.BLACK);
            double xScale = ((double) getWidth() - 2 * BORDER_GAP) / (dataRange[0][1] - dataRange[0][0]);
            double yScale = ((double) getHeight() - 2 * BORDER_GAP) / (dataRange[1][1] - dataRange[1][0]);

            for (int i = 0; i < data.length; i++) {
                int x = (int) (data[i].getFeatures()[0] * xScale + BORDER_GAP);
                int y = (int) (data[i].getFeatures()[1] * yScale + BORDER_GAP);
                g2.drawString(MARKERS[data[i].getLabel()], x, y);
            }
        }
    }

    private static int mostLikelyClass(float[] hist) {
        float maxCnt = hist[0];
        int mostLikelyClass = 0;
        for (int c = 1; c < hist.length; c++) {
            if (hist[c] > maxCnt) {
                maxCnt = hist[c];
                mostLikelyClass = c;
            }
        }
        return mostLikelyClass;
    }

    private static interface SampleGenerator {
        public DataPoint generate();
    }

    private static DataPoint[] generateData(int N, SampleGenerator sampleGenerator) {
        DataPoint[] data = new DataPoint[N];
        for (int i = 0; i < N; i++) {
            data[i] = sampleGenerator.generate();
        }
        return data;
    }

    private static class UniformSampleGenerator implements SampleGenerator {
        private final float[][][] sampleRanges;
        private final int nClasses;
        private final int featureDimentions;

        public UniformSampleGenerator(float[][][] sampleRanges) {
            this.sampleRanges = sampleRanges;
            nClasses = sampleRanges.length;
            featureDimentions = sampleRanges[0].length;
        }

        @Override
        public DataPoint generate() {
            int label = random.nextInt(nClasses);
            float[] features = new float[featureDimentions];
            for (int f = 0; f < featureDimentions; f++) {
                features[f] = sampleRanges[label][f][0]
                        + random.nextFloat() * (sampleRanges[label][f][1] - sampleRanges[label][f][0]);
            }
            return new DataPoint(label, features);
        }
    }

    private static class Spiral2DSampleGenerator implements SampleGenerator {

        private final int nClasses;

        public Spiral2DSampleGenerator(int nClasses) {
            this.nClasses = nClasses;
        }

        @Override
        public DataPoint generate() {
            int label = random.nextInt(nClasses);
            float r = (float) ((random.nextGaussian() + 3) * 2);
            double angle = 2 * Math.PI / nClasses * label + random.nextFloat() * 5 * Math.PI / nClasses / 10 + 0.5 * r;
            float[] features = new float[2];
            features[0] = (float) (r * Math.cos(angle) + 20);
            features[1] = (float) (r * Math.sin(angle) + 20);
            return new DataPoint(label, features);
        }
    }

}
