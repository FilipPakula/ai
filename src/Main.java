import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;

public class Main {
    public static void main(String[] args) throws IOException {
        double[] in = {2.0, 0.5, 1.0};
        Network network = new Network(3, 2,1, 3);
        double[] out = network.use(in);
        for(double d : out)
            System.out.println(d);
    }
}
