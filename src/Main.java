import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {
    static void main() throws IOException {

        double[] w = DataReader.readWeights("weights.csv");

        Neuron neuron = new Neuron(w, 0.1);

        ArrayList<Data> training_data = DataReader.read("training_data.csv");

        neuron.train(1000, training_data);

        Scanner scanner = new Scanner(System.in);
        double[] in = new double[3];

        System.out.print("Speed (km/h): "); in[0] = scanner.nextDouble();
        System.out.print("Lifting (kg): "); in[1] = scanner.nextDouble();
        System.out.print("Range (km): "); in[2] = scanner.nextDouble();

        Data test_data = new Data(in, 0.0);

        System.out.println();
        double fighterProbability = neuron.inference(test_data.input);
        String aircraftType = fighterProbability >= 0.5 ? "fighter" : "cargo";
        System.out.println("Prediction: " + aircraftType);
    }
}
