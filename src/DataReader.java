import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

public class DataReader {

    /** Reads comma-separated rows where all but the final number are inputs. */
    public static ArrayList<Data> read(String path) throws IOException {
        ArrayList<Data> data = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(path))) {
            String line;
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) {
                    continue;
                }

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

        return data;
    }

    /** Reads output weights from a CSV file (values may span multiple rows). */
    public static double[] readWeights(String path) throws IOException {
        ArrayList<Double> values = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(path))) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) {
                    continue;
                }

                for (String value : line.split(",", -1)) {
                    try {
                        values.add(Double.parseDouble(value.trim()));
                    } catch (NumberFormatException e) {
                        throw new IOException("Invalid weight at line " + lineNumber, e);
                    }
                }
            }
        }

        if (values.isEmpty()) {
            throw new IOException("No weights found in " + path);
        }

        double[] weights = new double[values.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = values.get(i);
        }
        return weights;
    }
}
