import java.util.ArrayList;

import static java.util.Collections.shuffle;

public class Network {
    Layer[] layers;
    double[] means;
    double[] standardDeviations;

    public Network(NetworkData n)
    {
        layers = new Layer[n.hiddenCount + 1];

        layers[0] = new Layer(n.inHidden, n.inputs, Func.RELU);

        for(int i = 1; i < n.hiddenCount; i++){
            layers[i] = new Layer(n.inHidden, n.inHidden, Func.RELU);
        }

        layers[n.hiddenCount] = new Layer(n.outputs, n.inHidden, n.outputType);
    }

    double use(double[] in)
    {
        double[] output = normalize(in);
        for(Layer layer : layers)
        {
            output = layer.use(output);
        }
        return output[0];
    }

    void train(ArrayList<Data> data)
    {
        for(Data row : data) {
// Forward pass: zapamiętaj aktywacje każdej warstwy.

            double[][] activations = new double[layers.length + 1][];
            activations[0] = normalize(row.input);

            for (int l = 0; l < layers.length; l++) {
                activations[l + 1] = layers[l].use(activations[l]);
            }
// Backpropagation: policz delty, zanim zmienisz jakiekolwiek wagi.

            double[][] deltas = new double[layers.length][];

            int last = layers.length - 1;
            deltas[last] = new double[layers[last].neurons.length];

            Neuron outputNeuron = layers[last].neurons[0];
            double prediction = activations[last + 1][0];

            deltas[last][0] =
                    2.0 * (prediction - row.target)
                            * outputNeuron.activationDerivativeFromLast();

            for (int l = last - 1; l >= 0; l--) {
                deltas[l] = new double[layers[l].neurons.length];

                for (int j = 0; j < layers[l].neurons.length; j++) {
                    double errorFromNextLayer = 0.0;

                    for (int k = 0; k < layers[l + 1].neurons.length; k++) {
                        errorFromNextLayer +=
                                deltas[l + 1][k] * layers[l + 1].neurons[k].weight[j];
                    }

                    Neuron neuron = layers[l].neurons[j];
                    deltas[l][j] =
                            errorFromNextLayer * neuron.activationDerivativeFromLast();
                }
            }

// Aktualizacja parametrów po obliczeniu delt całej sieci.

            for (int l = 0; l < layers.length; l++) {
                for (int j = 0; j < layers[l].neurons.length; j++) {
                    Neuron neuron = layers[l].neurons[j];

                    for (int i = 0; i < neuron.weight.length; i++) {
                        neuron.weight[i] -= neuron.rate * deltas[l][j] * activations[l][i];
                    }

                    neuron.bias -= neuron.rate * deltas[l][j];
                }
            }
        }
    }

    void training(int x, ArrayList<Data> data)
    {
        fitNormalization(data);

        for(int i = 0; i < x; i++)
        {
            train(data);
            shuffle(data);
        }
    }

    void fitNormalization(ArrayList<Data> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Dane treningowe nie mogą być puste");
        }

        int featureCount = data.getFirst().input.length;
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
