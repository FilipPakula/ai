import java.util.ArrayList;
import java.util.Random;
import static java.lang.Math.tanh;

public class Neuron {
    double bias, rate;
    int inputs;
    double[] weight;
    Func func;

    public Neuron(int inputs,double rate, Func f) {
        this.bias = 0.0;
        this.rate = rate;
        this.inputs = inputs;
        this.func = f;
        this.weight = new double[inputs];

        Random random = new Random();

        for(int i = 0; i < inputs; i++)
        {
            this.weight[i] = random.nextDouble() - 0.5;
        }
    }

    public static double sigmoid(double x) {return 1.0 / (1.0 + Math.exp(-x));}

    public static double relu(double x){return x > 0 ? x : 0;}

    public void epoch(Data data)
    {
        double z = weightedSum(data.input);
        double prediction = activate(z);

        double gradient = 2.0 * (prediction - data.target) * activationDerivative(z, prediction);

        for (int i = 0; i < inputs; i++) {
            this.weight[i] -= this.rate * gradient * data.input[i];
        }

        this.bias -= this.rate * gradient;
    }

    public double inference(double[] in) {
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
        return switch (this.func) {
            case SIGMOID -> sigmoid(x);
            case TANH -> tanh(x);
            case RELU -> relu(x);
        };
    }

    private double activationDerivative(double x, double activated) {
        return switch (func) {
            case SIGMOID -> activated * (1.0 - activated);
            case TANH -> 1.0 - activated * activated;
            case RELU -> x > 0.0 ? 1.0 : 0.0;
        };
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
