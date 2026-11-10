public class Neuron {
    double weight, bias, target, rate;

    public Neuron() {
        this.weight = 0.5;
        this.bias = 0.0;
        this.target = 1.0;
        this.rate = 0.1;
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public Neuron train(double x)
    {
        double z = x * this.weight + this.bias;
        double prediction = sigmoid(z);

        double loss = (prediction - this.target) * (prediction - this.target);
    }

}
