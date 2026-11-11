public class Neuron {
    double weight, bias, target, rate, in, loss, prediction;

    public Neuron() {
        this.weight = 0.5;
        this.bias = 0.0;
        this.target = 1.0;
        this.rate = 0.1;
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public void epoch()
    {
        double z = this.in * this.weight + this.bias;
        double prediction = sigmoid(z);

        this.loss = (prediction - this.target) * (prediction - this.target);
        double gradient = 2 * (prediction - target) * prediction * (1 - prediction) * this.in;
        double bias_gradient = 2 * (prediction - target) * prediction * (1 - prediction);

        this.weight -= this.rate * gradient;
        this.bias -= this.rate * bias_gradient;
        this.prediction = prediction;
    }

    public void train(int x, double in){
        this.in = in;
        for( int i = 0; i < x; i++ ) {
            this.epoch();
        }
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

}
