import java.io.IOException;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) throws IOException {

        Neuron neuron = new Neuron();

        ArrayList<Data> training_data = DataReader.read("training_data.csv");
        ArrayList<Data> test_data = DataReader.read("test_data.csv");

        neuron.train(1000, training_data);

        double loss = neuron.evaluate(test_data);
        System.out.println(loss);
    }
}