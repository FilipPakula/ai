import java.io.IOException;
import java.util.ArrayList;

public class Main {
    static void main() throws IOException {

        Network network = new Network(3, 2, 3);

        ArrayList<Data> data = DataReader.read("training_data.csv");
        network.training(10000, data);

        //double[] in = {2.0, 0.5, 1.0};
        double[] in = {0.6, 50.0, 10.0};

        System.out.println(network.use(in));
    }
}
