import java.io.IOException;
import java.util.ArrayList;

public class Main {
    static void main() throws IOException {

        NetworkData netData = new NetworkData();
        netData.setHiddenCount(2);
        netData.setInHidden(3);
        netData.setInputs(3);
        netData.setOutputType(Func.SIGMOID);
        netData.setOutputs(1);

        Network network = new Network(netData);

        ArrayList<Data> data = DataReader.read("training_data.csv");
        network.training(10000, data);

        //double[] in = {2.0, 0.5, 1.0};
        double[] in = {0.6, 50.0, 10.0};

        System.out.println(network.use(in));
    }
}
