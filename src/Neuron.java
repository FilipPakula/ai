import java.util.ArrayList;

public class Neuron {
    double bias, rate;
    int inputs;
    double[] weight;
    private double[] means;
    private double[] standardDeviations;

    public Neuron(int inputs, double weights, double rate) {
        if (inputs <= 0) throw new IllegalArgumentException("Number of inputs must be positive");
        if (rate <= 0) throw new IllegalArgumentException("Learning rate must be positive");
        this.bias = 0.0;
        this.rate = rate;
        this.inputs = inputs;

        this.weight = new double[inputs];

        for(int i = 0; i < inputs; i++)
        {
            this.weight[i] = weights;
        }
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public void epoch(Data data)
    {
        if (means == null) throw new IllegalStateException("Train the neuron before updating it");
        checkInput(data.input);
        checkTarget(data.target);
        double[] normalized = normalize(data.input);
        double prediction = predictNormalized(normalized);

        for (int i = 0; i < inputs; i++) {
            this.weight[i] -= this.rate * (prediction - data.target) * normalized[i];
        }

        this.bias -= this.rate * (prediction - data.target);
    }

    public void train(int x, ArrayList<Data> data){
        if (x <= 0) throw new IllegalArgumentException("Epoch count must be positive");
        if (data.isEmpty()) throw new IllegalArgumentException("Training data must not be empty");
        calculateNormalization(data);
        for(int i = 0; i < x; i++) {
            for (Data datum : data) {
                this.epoch(datum);
            }
        }
    }

    public double inference(double[] in) {
        checkInput(in);
        if (means == null) throw new IllegalStateException("Train the neuron before inference");
        return predictNormalized(normalize(in));
    }

    private double predictNormalized(double[] in) {
        double y = 0;
        for (int i = 0; i < inputs; i++) {
            y += in[i] * this.weight[i];
        }
        y += this.bias;

        return sigmoid(y);
    }

    private void calculateNormalization(ArrayList<Data> data) {
        means = new double[inputs];
        standardDeviations = new double[inputs];
        for (Data datum : data) {
            checkInput(datum.input);
            checkTarget(datum.target);
            for (int i = 0; i < inputs; i++) means[i] += datum.input[i];
        }
        for (int i = 0; i < inputs; i++) means[i] /= data.size();
        for (Data datum : data) {
            for (int i = 0; i < inputs; i++) {
                double difference = datum.input[i] - means[i];
                standardDeviations[i] += difference * difference;
            }
        }
        for (int i = 0; i < inputs; i++) {
            standardDeviations[i] = Math.sqrt(standardDeviations[i] / data.size());
            if (standardDeviations[i] < 1e-12) standardDeviations[i] = 1.0;
        }
    }

    private double[] normalize(double[] in) {
        double[] normalized = new double[inputs];
        for (int i = 0; i < inputs; i++) {
            normalized[i] = (in[i] - means[i]) / standardDeviations[i];
        }
        return normalized;
    }

    private void checkInput(double[] in) {
        if (in.length != inputs) {
            throw new IllegalArgumentException("Expected " + inputs + " inputs but got " + in.length);
        }
    }

    private void checkTarget(double target) {
        if (target != 0.0 && target != 1.0) {
            throw new IllegalArgumentException("Target must be 0 or 1");
        }
    }

    public double evaluate(ArrayList<Data> data)
    {
        double loss = 0;
        for(Data datum : data)
        {
            double difference = inference(datum.input) - datum.target;
            loss += difference * difference;
        }

        if(!data.isEmpty())return loss / data.size();
        else return 0;
    }
}
