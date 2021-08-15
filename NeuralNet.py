import math
import random


class Neuron:
    inputs: int
    weights: list[float]
    b: float

    @staticmethod
    def sigmoid(val: float) -> float:
        if val > 100:
            return 1
        if val < -100:
            return 0
        return 1 / (1 + math.exp(-val))

    def __init__(self, inputs: int):
        self.inputs = inputs
        self.weights = []
        while len(self.weights) < self.inputs:
            self.weights.append(random.random() * 20 - 10)
        self.b = random.random() * 20 - 10

    def push(self, input_data: list[float]) -> float:
        return Neuron.sigmoid(
            sum(
                map(
                    lambda it: self.weights[it] * input_data[it],
                    range(self.inputs)
                )
            ) + self.b
        )
    
    @staticmethod
    def from_array(inputs: int, arr: list[float]):
        neuron = Neuron(inputs)
        neuron.weights = arr[:-1]
        neuron.b = arr[-1]
        return neuron


class NeuralNet:
    inputs: int
    outputs: int
    hidden: list[int]
    layers: list[list[Neuron]]
    
    def __init__(self, inputs: int, outputs: int, hidden: list[int]):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden
        self.layers = []
        
        current_inputs = self.inputs
        for hid in self.hidden:
            new_layer = []
            while len(new_layer) < hid:
                new_layer.append(Neuron(current_inputs))
                
            self.layers.append(new_layer)
            current_inputs = hid
        
        output_layer = []
        while len(output_layer) < self.outputs:
            output_layer.append(Neuron(current_inputs))
        self.layers.append(output_layer)
    
    def push(self, input_data: list[float]) -> list[float]:
        current_data = input_data[:]
        for layer in self.layers:
            current_data = list(
                map(
                    lambda neuron: neuron.push(current_data),
                    layer
                )
            )
        
        return current_data
    
    def json(self) -> dict:
        weights = []
        for layer in self.layers:
            for neuron in layer:
                for w in neuron.weights:
                    weights.append(w)
                weights.append(neuron.b)
        return {
            'inputs': self.inputs,
            'outputs': self.outputs,
            'hidden': self.hidden,
            'weights': weights
        }
    
    @staticmethod
    def from_json(data: json):
        neural_net = NeuralNet(data['inputs'], data['outputs'], data['hidden'])
        from_it = 0
        current_inputs = data['inputs']
        for layer in neural_net.layers:
            for i in range(len(layer)):
                to_it = from_it + layer[i].inputs + 1
                layer[i] = Neuron.from_array(current_inputs, data['weights'][from_it:to_it])
                from_it = to_it

            current_inputs = len(layer)

        return neural_net
