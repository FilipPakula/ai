import java.util.ArrayList;
import java.util.Objects;

import static java.lang.Math.tanh;

public class Neuron {
    double bias, rate;
    int inputs;
    double[] weight;
    private double[] means;
    private double[] standardDeviations;
    Func func;

    public Neuron(int inputs, double weights, double rate, Func f) {
        this.bias = 0.0;
        this.rate = rate;
        this.inputs = inputs;
        this.func = f;
        this.weight = new double[inputs];

        for(int i = 0; i < inputs; i++)
        {
            this.weight[i] = weights;
        }
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double relu(double x){
        return x > 0 ? x : 0;
    }

    public void epoch(Data data)
    {
        double[] normalized = normalize(data.input);
        double z = weightedSum(normalized);
        double prediction = activate(z);

        double gradient = 2.0 * (prediction - data.target) * activationDerivative(z, prediction);

        for (int i = 0; i < inputs; i++) {
            this.weight[i] -= this.rate * gradient * normalized[i];
        }

        this.bias -= this.rate * gradient;
    }

    public void train(int x, ArrayList<Data> data){
        calculateNormalization(data);
        for(int i = 0; i < x; i++) {
            for (Data datum : data) {
                this.epoch(datum);
            }
        }
    }

    public double inference(double[] in) {
        in = normalize(in);
        return activate(weightedSum(in));
    }

    private double weightedSum(double[] in) {
        double y = 0;
        for (int i = 0; i < inputs; i++) {
            y += in[i] * this.weight[i];
        }
        return y + this.bias;
    }

    private double activate(double x) {
        switch(this.func)
        {
            case SIGMOID: return sigmoid(x);
            case TANH: return tanh(x);
            case RELU: return relu(x);
            default: throw new IllegalStateException("Unsupported activation: " + func);
        }
    }

    private double activationDerivative(double x, double activated) {
        switch (func) {
            case SIGMOID: return activated * (1.0 - activated);
            case TANH: return 1.0 - activated * activated;
            case RELU: return x > 0.0 ? 1.0 : 0.0;
            default: throw new IllegalStateException("Unsupported activation: " + func);
        }
    }

    private void calculateNormalization(ArrayList<Data> data) {
        means = new double[inputs];
        standardDeviations = new double[inputs];
        for (Data datum : data) {
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
