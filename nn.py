from backprop import Value
import random

# nin = number of inputs
class Neuron:
    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # weights
        self.b = Value(random.uniform(-1, 1)) # bias

    def __call__(self, x):
        # w * x + b -> forward pass
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.tanh()
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] # a layer is just a list of neurons

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # return [p for neuron in self.neurons for p in neuron.parameters()]
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
        
# multi layer perceptron
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # nouts is a list of layers containing neurons
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def parameters(self):
        # return [p for layer in self.layers for p in layer.parameters()]
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params