public class Data {
    double[] input;
    double target;

    public Data(double[] in, double target)
    {
        input = new double[in.length];
        for(int i = 0; i < in.length; i++) this.input[i] = in[i];

        this.target = target;
    }
}
