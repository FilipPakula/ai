import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Neuron {
    double bias;
    double rate;
    double[] weight;
    private double[] mean;
    private double[] standardDeviation;
    private boolean trained;

    public Neuron(double[] initialWeights, double rate) {
        if (initialWeights.length == 0) {
            throw new IllegalArgumentException("At least one weight is required");
        }
        if (rate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }

        this.rate = rate;
        this.weight = initialWeights.clone();

        // Keep the relative values from weights.csv, but start out of sigmoid saturation.
        double maxAbsWeight = 0.0;
        for (double value : weight) {
            maxAbsWeight = Math.max(maxAbsWeight, Math.abs(value));
        }
        if (maxAbsWeight > 0.0) {
            for (int i = 0; i < weight.length; i++) {
                weight[i] = 0.1 * weight[i] / maxAbsWeight;
            }
        }
    }

    public static double sigmoid(double x) {
        if (x >= 0) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        double exp = Math.exp(x);
        return exp / (1.0 + exp);
    }

    public void train(int epochs, ArrayList<Data> data) {
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epoch count must be positive");
        }
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Training data must not be empty");
        }

        for (Data datum : data) {
            validateTarget(datum.target);
        }
        fitNormalizer(data);
        trained = true;
        ArrayList<Data> shuffledData = new ArrayList<>(data);
        Random random = new Random(42);

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(shuffledData, random);
            for (Data datum : shuffledData) {
                double[] normalized = normalize(datum.input);
                double prediction = predictNormalized(normalized);

                // Binary cross-entropy with sigmoid has dL/dz = prediction - target.
                double error = prediction - datum.target;
                for (int i = 0; i < weight.length; i++) {
                    weight[i] -= rate * error * normalized[i];
                }
                bias -= rate * error;
            }
        }
    }

    public double inference(double[] input) {
        ensureTrained();
        return predictNormalized(normalize(input));
    }

    public double evaluate(ArrayList<Data> data) {
        if (data.isEmpty()) {
            return 0.0;
        }
        double totalLoss = 0.0;
        for (Data datum : data) {
            validateTarget(datum.target);
            double prediction = inference(datum.input);
            totalLoss += (prediction - datum.target) * (prediction - datum.target);
        }
        return totalLoss / data.size();
    }

    private void fitNormalizer(ArrayList<Data> data) {
        mean = new double[weight.length];
        standardDeviation = new double[weight.length];

        for (Data datum : data) {
            checkInputSize(datum.input);
            for (int i = 0; i < weight.length; i++) {
                mean[i] += datum.input[i];
            }
        }
        for (int i = 0; i < weight.length; i++) {
            mean[i] /= data.size();
        }
        for (Data datum : data) {
            for (int i = 0; i < weight.length; i++) {
                double difference = datum.input[i] - mean[i];
                standardDeviation[i] += difference * difference;
            }
        }
        for (int i = 0; i < weight.length; i++) {
            standardDeviation[i] = Math.sqrt(standardDeviation[i] / data.size());
            if (standardDeviation[i] < 1e-12) {
                standardDeviation[i] = 1.0;
            }
        }
    }

    private double[] normalize(double[] input) {
        ensureTrained();
        checkInputSize(input);
        double[] normalized = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            normalized[i] = (input[i] - mean[i]) / standardDeviation[i];
        }
        return normalized;
    }

    private double predictNormalized(double[] normalizedInput) {
        double z = bias;
        for (int i = 0; i < weight.length; i++) {
            z += normalizedInput[i] * weight[i];
        }
        return sigmoid(z);
    }

    private void checkInputSize(double[] input) {
        if (input.length != weight.length) {
            throw new IllegalArgumentException("Expected " + weight.length
                    + " input features but got " + input.length);
        }
    }

    private void validateTarget(double target) {
        if (target != 0.0 && target != 1.0) {
            throw new IllegalArgumentException("Target must be 0 (cargo) or 1 (fighter), got " + target);
        }
    }

    private void ensureTrained() {
        if (!trained) {
            throw new IllegalStateException("Train the model before inference");
        }
    }
}
