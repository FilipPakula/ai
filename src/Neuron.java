import java.util.ArrayList;

public class Neuron {
    double weight, bias, rate, loss, prediction;

    public Neuron() {
        this.weight = 0.5;
        this.bias = 0.0;
        this.rate = 0.1;
        this.loss = 0.0;
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public void epoch(double in, double target)
    {
        double prediction = inference(in);

        this.loss += (prediction - target) * (prediction - target);
        double gradient = 2 * (prediction - target) * prediction * (1 - prediction) * in;
        double bias_gradient = 2 * (prediction - target) * prediction * (1 - prediction);

        this.weight -= this.rate * gradient;
        this.bias -= this.rate * bias_gradient;
        this.prediction = prediction;
    }

    public void train(int x, ArrayList<Data> data){
        for(int i = 0; i < x; i++) {
            for (Data datum : data) {
                this.epoch(datum.input, datum.target);
            }
        }
        this.loss /= x * data.size();
    }

    public void printInfo()
    {
        System.out.println("weight: " + this.weight);
        System.out.println("bias: " + this.bias);
        System.out.println("loss: " + this.loss);
        System.out.println("prediction: " + this.prediction);
    }

    public double inference(double in)
    {
        return sigmoid(in * this.weight + this.bias);
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
