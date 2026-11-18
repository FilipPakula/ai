import java.util.ArrayList;

public class Network {
    Layer[] layers;
    double[] means;
    double[] standardDeviations;

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
        double[] outputs = normalize(in);
        for(Layer layer : layers)
        {
            outputs = layer.use(outputs);
        }
        return outputs;
    }

    void train(ArrayList<Data> data)
    {

    }

    void training(int x, ArrayList<Data> data)
    {
        fitNormalization(data);

        for(int i = 0; i < x; i++)
        {
            train(data);
        }
    }

    void fitNormalization(ArrayList<Data> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Dane treningowe nie mogą być puste");
        }

        int featureCount = data.get(0).input.length;
        means = new double[featureCount];
        standardDeviations = new double[featureCount];

        // Średnia każdej cechy
        for (Data row : data) {
            if (row.input.length != featureCount) {
                throw new IllegalArgumentException("Niezgodna liczba cech");
            }
            for (int j = 0; j < featureCount; j++) {
                means[j] += row.input[j];
            }
        }
        for (int j = 0; j < featureCount; j++) {
            means[j] /= data.size();
        }

        // Odchylenie standardowe każdej cechy
        for (Data row : data) {
            for (int j = 0; j < featureCount; j++) {
                double difference = row.input[j] - means[j];
                standardDeviations[j] += difference * difference;
            }
        }
        for (int j = 0; j < featureCount; j++) {
            standardDeviations[j] =
                    Math.sqrt(standardDeviations[j] / data.size());

            if (standardDeviations[j] < 1e-12) {
                standardDeviations[j] = 1.0;
            }
        }
    }

    private double[] normalize(double[] in) {
        double[] normalized = new double[in.length];
        for (int i = 0; i < in.length; i++) {
            normalized[i] = (in[i] - means[i]) / standardDeviations[i];
        }
        return normalized;
    }
}
