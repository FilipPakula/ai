import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public class DataReader {
    public static ArrayList<Data> read(String path) throws IOException {
        ArrayList<Data> data = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(path))) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.trim().isEmpty()) continue;

                String[] values = line.split(",", -1);
                if (values.length < 2) {
                    throw new IOException("Invalid data at line " + lineNumber
                            + ": expected at least one input and a target");
                }

                double[] inputs = new double[values.length - 1];
                try {
                    for (int i = 0; i < inputs.length; i++) {
                        inputs[i] = Double.parseDouble(values[i].trim());
                    }
                    double target = Double.parseDouble(values[values.length - 1].trim());
                    data.add(new Data(inputs, target));
                } catch (NumberFormatException e) {
                    throw new IOException("Invalid number at line " + lineNumber, e);
                }
            }
        }

        if (data.isEmpty()) throw new IOException("No training data found in " + path);
        int expectedInputs = data.get(0).input.length;
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).input.length != expectedInputs) {
                throw new IOException("Inconsistent number of inputs at data row " + (i + 1));
            }
        }
        return data;
    }
}
