public class Main {
    public static void main(String[] args) {

        Neuron neuron = new Neuron();

        neuron.train(1000, 0.5);

        neuron.printInfo();

        System.out.println();
        System.out.println("Test 0: " + neuron.inference(0));
        System.out.println("Test 1: " + neuron.inference(1));
        System.out.println("Test 2: " + neuron.inference(2));
        System.out.println("Test 5: " + neuron.inference(5));
    }
}