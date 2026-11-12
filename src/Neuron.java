import java.util.ArrayList;

public class Neuron {
    double bias, rate, loss, prediction;
    double[] weight;

    public Neuron(double[] weights, double rate) {
        this.bias = 0.0;
        this.rate = rate;
        this.loss = 0.0;

        this.weight = weights.clone();
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public void epoch(double[] in, double target)
    {
        if (in.length != weight.length) {
            throw new IllegalArgumentException("Expected " + weight.length
                    + " inputs but got " + in.length);
        }

        double prediction = 0;
        double[] gradient = new double[in.length];

        for (int i = 0; i < in.length; i++) {
            prediction += in[i] * this.weight[i];
        }

        prediction = sigmoid(prediction + this.bias);

        for (int i = 0; i < in.length; i++) {
            gradient[i] = 2 * (prediction - target) * prediction * (1 - prediction) * in[i];
            this.weight[i] -= this.rate * gradient[i];
        }

        double bias_gradient = 2 * (prediction - target) * prediction * (1 - prediction);
        this.bias -= this.rate * bias_gradient;
    }

    public void train(int x, ArrayList<Data> data){
        for(int i = 0; i < x; i++) {
            for (Data datum : data) {
                this.epoch(datum.input, datum.target);
            }
        }
    }

    public double inference(double[] in)
    {
        if (in.length != weight.length) {
            throw new IllegalArgumentException("Expected " + weight.length
                    + " inputs but got " + in.length);
        }

        double y = 0;
        for (int i = 0; i < in.length; i++) {
            y += in[i] * this.weight[i];
        }
        return sigmoid(y + this.bias);
    }

    public double evaluate(ArrayList<Data> data)
    {
        double loss = 0;
        for(Data datum : data)
        {
            loss += (inference(datum.input) - datum.target) * (inference(datum.input) - datum.target);
        }

        if(!data.isEmpty())return loss / data.size();
        else return 0;
    }
}
