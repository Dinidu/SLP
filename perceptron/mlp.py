from numpy import exp, array, random, dot


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self,iterations,hidden_layers,neurons_in_hidden_layer):
        self.number_of_training_iterations = iterations
        self.number_of_hidden_layers = hidden_layers
        self.number_of_neurons_in_hidden_layer = neurons_in_hidden_layer
        self.weightVectors = []
        pass

    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs):
        self.weightVectors = []
        for layer in range(self.number_of_hidden_layers) :
            if not self.weightVectors:
                weights = NeuronLayer(len(training_set_inputs[0]),self.number_of_neurons_in_hidden_layer)
                self.weightVectors.append(weights)
            elif len(self.weightVectors) == self.number_of_hidden_layers :
                # TODO : assumption is output layer is having only one neuron
                weights = NeuronLayer(self.number_of_neurons_in_hidden_layer,1)
                self.weightVectors.append(weights)
            else:
                # TODO : assumption is each hidden layer is having similar numberof neurons
                weights = NeuronLayer(self.number_of_neurons_in_hidden_layer,self.number_of_neurons_in_hidden_layer)
                self.weightVectors.append(weights)

        for iteration in range(self.number_of_training_iterations):
            #forward pass
            # Pass the training set through our neural network
            outputVectors = []
            for layer in range(self.number_of_hidden_layers):
                if not outputVectors:
                    inputs = training_set_inputs
                else :
                    inputs = outputVectors[layer-1]
                weight = self.weightVectors[layer-1]
                ithLayerOutput = self.getLayerOutput(inputs, weight)
                outputVectors.append(ithLayerOutput)

            finaloutput = self.getLayerOutput(outputVectors[self.number_of_hidden_layers],self.weightVectors[self.number_of_hidden_layers])
            outputVectors.append(finaloutput)

            errorVectors = []
            gradientVectors = []

            # # Calculate the error for each layer (The difference between the desired output and the predicted output).
            for layer in range(self.number_of_hidden_layers):
                if not errorVectors:
                    error_vector = training_set_outputs - outputVectors[self.number_of_hidden_layers]
                    errorVectors.append(error_vector)
                    gradient_vector = error_vector * self.__sigmoid_derivative(outputVectors[self.number_of_hidden_layers])
                    gradientVectors.append(gradient_vector)
                elif len(errorVectors) == self.number_of_hidden_layers:
                    error_vector = gradientVectors[self.number_of_hidden_layers] * self.__sigmoid_derivative(outputVectors[0])
                    errorVectors.append(error_vector)
                    gradient_vector = error_vector * self.__sigmoid_derivative(outputVectors[self.number_of_hidden_layers])
                    gradientVectors.append(gradient_vector)
                else :
                    error_vector = gradientVectors[layer-1].dot(self.weightVectors[self.number_of_hidden_layers-layer].T)
                    errorVectors.append(error_vector)
                    gradient_vector = error_vector * self.__sigmoid_derivative(outputVectors[self.number_of_hidden_layers-layer])
                    gradientVectors.append(gradient_vector)

            for layer in range(self.number_of_hidden_layers):
                # Calculate how much to adjust the weights by
                if layer ==0 :
                    layer_adjustment = training_set_inputs.T.dot(gradientVectors[self.number_of_hidden_layers-layer])
                    # Adjust the weights.
                    self.weightVectors[layer] += layer_adjustment
                else :
                    layer_adjustment = outputVectors[layer].T.dot(gradientVectors[self.number_of_hidden_layers - layer])
                    # Adjust the weights.
                    self.weightVectors[layer] += layer_adjustment

    # The neural network output.
    def getLayerOutput(self, inputs,weights):
        output = self.__sigmoid(dot(inputs, weights))
        return output

if __name__ == "__main__":

    neural_network = NeuralNetwork(2,3,4)

    training_set_inputs = array([[0, 0, 1],
                                 [0, 1, 1],
                                 [1, 0, 1],
                                 [0, 1, 0],
                                 [1, 0, 0],
                                 [1, 1, 1],
                                 [0, 0, 0]])

    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    neural_network.train(training_set_inputs, training_set_outputs)

    # # Test the neural network with a new situation.
    output = neural_network.getLayerOutput(array([1, 1, 0]))
    print output