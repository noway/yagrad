import cmath
import random
from functools import reduce
from graphlib import TopologicalSorter

grad_add = lambda node: [node.grad, node.grad]
grad_exp = lambda node: [cmath.exp(node._deps[0].data) * node.grad]
grad_log = lambda node: [node.grad / node._deps[0].data]

op_add = lambda a, b: a.data + b.data
op_exp = lambda a: cmath.exp(a.data)
op_log = lambda a: cmath.log(a.data)

add = lambda a, b: Value(op_add(a, b), (a, b), op_add, grad_add)
exp = lambda a: Value(op_exp(a), (a,), op_exp, grad_exp)
log = lambda a: Value(op_log(a), (a,), op_log, grad_log)

sum = lambda xs: reduce(lambda b, x: add(b, x), xs, Value(0))
mul = lambda a, b: exp(add(log(a), log(b)))
pow = lambda a, b: exp(mul(log(a), b))
exp2x = lambda x: exp(mul(x, Value(2)))
tanh = lambda x: mul(add(exp2x(x), Value(-1)), pow(add(exp2x(x), Value(1)), Value(-1)))

module_params = lambda m: [p for module in m.modules for p in params(module)]
params = lambda m: [m] if isinstance(m, Value) else module_params(m)


class Value:
    def __init__(self, data, _deps=(), _op=None, _calcgrad=lambda _: []):
        self.data = data
        self._deps = _deps
        self._op = _op
        self._calcgrad = _calcgrad

    def items(self):
        return [(self, self._deps)] + [items for d in self._deps for items in d.items()]


class Neuron:
    def __init__(self, input_len):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(input_len)]
        self.b = Value(random.uniform(-1, 1))
        self.modules = self.w + [self.b]

    def forward(self, x):
        return tanh(sum([mul(wi, xi) for wi, xi in zip(self.w, x)] + [self.b]))


class Layer:
    def __init__(self, nin, nout):
        self.modules = [Neuron(nin) for _ in range(nout)]

    def forward(self, x):
        return [neuron.forward(x) for neuron in self.modules]


class MultiLayerPerceptron:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.modules = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def forward(self, x):
        return reduce(lambda x, layer: layer.forward(x), self.modules, x)


def fit(n, loss, epochs, learning_rate):
    wnb = params(n)
    graph = list(TopologicalSorter(loss).static_order())
    for k in range(epochs):  # mutates the fixed graph
        for node in graph:
            node.grad = 0  # zero grad
            if node._op:
                node.data = node._op(*node._deps)  # forward
        graph[-1].grad = 1  # set grad of loss
        for node in reversed(graph):  # backward
            for i, grad in enumerate(node._calcgrad(node)):
                node._deps[i].grad += grad  # auto grad
        for p in wnb:
            p.data -= learning_rate * p.grad  # optim gd
        print(k, loss.data)


n = MultiLayerPerceptron(3, [4, 4, 1])
xs = [
    [Value(2), Value(3), Value(-1)],
    [Value(3), Value(-1), Value(0.5)],
    [Value(0.5), Value(1), Value(1)],
    [Value(1), Value(1), Value(-1)],
]
ys = [Value(1), Value(-1), Value(-1), Value(1)]

ypred = [n.forward(x)[0] for x in xs]
loss = sum(pow(add(yout, mul(y, Value(-1))), Value(2)) for y, yout in zip(ys, ypred))
fit(n, loss, epochs=500, learning_rate=0.1)
ypred_final = [n.forward(x)[0] for x in xs]
print([yout.data.real for yout in ypred_final])
