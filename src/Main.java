import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;

public class Main {
    public static void main(String[] args) throws IOException {
        ArrayList<Data> trainingData = DataReader.read("training_data.csv");
        Neuron neuron = new Neuron(3, 0.1, 1.0);

        neuron.train(100000, trainingData);

        double[] example = {2.5, 7.0, 4.0};
        //double[] example = {0.6, 70.0, 10.0};

        System.out.printf(Locale.ROOT, "Prediction: %.12f%n", neuron.inference(example));
        System.out.printf(Locale.ROOT, "Mean squared error on training data: %.12f%n",
                neuron.evaluate(trainingData));
    }
}
