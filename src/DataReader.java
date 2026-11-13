import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class DataReader {

    public static ArrayList<Data> read(String path) throws IOException {
        ArrayList<Data> data = new ArrayList<>();

        BufferedReader reader = new BufferedReader(new FileReader(path));

        String line;

        while ((line = reader.readLine()) != null) {
            String[] values = line.split(",");

            double input = Double.parseDouble(values[0]);
            double target = Double.parseDouble(values[1]);

            data.add(new Data(input, target));
        }

        reader.close();

        return data;
    }
}