import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

public class Main {
    Random random = new Random();
    public static void main(String[] args) throws IOException {
        double[] in = {2.0, 0.5, 1.0};
        Network network = new Network(3, 2,1, 3);
        double[] out = network.use(in);
        for(double d : out)
            System.out.println(d);
    }
}
