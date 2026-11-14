public class Layer {
    private final Neuron[] neurons;

    public Layer(int neuronCount, int inputCount, Func func) {
        neurons = new Neuron[neuronCount];

        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputCount, 0.1, 0.1, func);
        }
    }

    public double[] use(double[] inputs) {
        double[] outputs = new double[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].inference(inputs);
        }

        return outputs;
    }
}