public class Network {
    Layer[] layers;

    public Network(int inputs, int hiddenCount, int outputs, int inHidden)
    {
        layers = new Layer[hiddenCount + 1];

        layers[0] = new Layer(inHidden, inputs, Func.RELU);

        for(int i = 1; i < hiddenCount; i++){
            layers[i] = new Layer(inHidden, inHidden, Func.RELU);
        }

        layers[hiddenCount] = new Layer(outputs, inHidden, Func.SIGMOID);
    }

    double[] use(double[] in)
    {
        double[] outputs = in;
        for(Layer layer : layers)
        {
            outputs = layer.use(outputs);
        }
        return outputs;
    }
}
