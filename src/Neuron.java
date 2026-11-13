import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Neuron {
    private final double[] weights;
    private final double learningRate;
    private double bias;
    private double[] means;
    private double[] standardDeviations;
    private boolean trained;

    public Neuron(double[] initialWeights, double learningRate) {
        this.weights = initialWeights.clone();
        this.learningRate = learningRate;

        // Scale starting weights down so the sigmoid does not start saturated.
        double largestWeight = 0;
        for (double weight : weights) {
            largestWeight = Math.max(largestWeight, Math.abs(weight));
        }
        if (largestWeight > 0) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] = 0.1 * weights[i] / largestWeight;
            }
        }
    }

    public void train(int epochs, ArrayList<Data> trainingData) {
        for (Data row : trainingData) {
            checkInputSize(row.input);
            checkTarget(row.target);
        }
        calculateNormalization(trainingData);
        trained = true;

        // Shuffle each epoch so examples from one class are not always seen together.
        ArrayList<Data> rows = new ArrayList<>(trainingData);
        Random random = new Random(42);
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(rows, random);
            for (Data row : rows) {
                double[] input = normalize(row.input);
                double prediction = predictNormalized(input);
                double error = prediction - row.target;

                // For sigmoid with binary cross-entropy, the gradient is prediction - target.
                for (int i = 0; i < weights.length; i++) {
                    weights[i] -= learningRate * error * input[i];
                }
                bias -= learningRate * error;
            }
        }
    }

    public double inference(double[] input) {
        if (!trained) {
            throw new IllegalStateException("Train the model before inference");
        }
        checkInputSize(input);
        return predictNormalized(normalize(input));
    }

    public double evaluate(ArrayList<Data> data) {
        if (data.isEmpty()) {
            return 0;
        }

        double squaredError = 0;
        for (Data row : data) {
            checkTarget(row.target);
            double difference = inference(row.input) - row.target;
            squaredError = difference * difference;
        }
        return squaredError / data.size();
    }

    private void calculateNormalization(ArrayList<Data> data) {
        means = new double[weights.length];
        standardDeviations = new double[weights.length];

        for (Data row : data) {
            for (int i = 0; i < weights.length; i++) {
                means[i] += row.input[i];
            }
        }
        for (int i = 0; i < weights.length; i++) {
            means[i] /= data.size();
        }

        for (Data row : data) {
            for (int i = 0; i < weights.length; i++) {
                double difference = row.input[i] - means[i];
                standardDeviations[i] += difference * difference;
            }
        }
        for (int i = 0; i < weights.length; i++) {
            standardDeviations[i] = Math.sqrt(standardDeviations[i] / data.size());
            if (standardDeviations[i] < 1e-12) {
                standardDeviations[i] = 1;
            }
        }
    }

    private double[] normalize(double[] input) {
        double[] normalized = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            normalized[i] = (input[i] - means[i]) / standardDeviations[i];
        }
        return normalized;
    }

    private double predictNormalized(double[] input) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += input[i] * weights[i];
        }
        return 1.0 / (1.0 + Math.exp(-sum));
    }

    private void checkInputSize(double[] input) {
        if (input.length != weights.length) {
            throw new IllegalArgumentException("Expected " + weights.length
                    + " input features but got " + input.length);
        }
    }

    private void checkTarget(double target) {
        if (target != 0 && target != 1) {
            throw new IllegalArgumentException("Target must be 0 (cargo) or 1 (fighter), got " + target);
        }
    }
}
