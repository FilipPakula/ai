import java.io.IOException;
import java.util.ArrayList;

public class Main {
    static void main() throws IOException {

        double[] w = DataReader.readWeights("weights.csv");

        Neuron neuron = new Neuron(w, 0.1);

        ArrayList<Data> training_data = DataReader.read("training_data.csv");

        neuron.train(1000, training_data);

        double[] in = {2.0, 8.0, 3.0};
        Data test_data = new Data(in, 0.0);

        System.out.println(neuron.inference(test_data.input));

    }
}
