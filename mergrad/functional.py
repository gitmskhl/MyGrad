import math
from mergrad.value import Value


def sin(value):
    value = Value.toValue(value)
    out = Value(math.sin(value.data), parents=(value,), op='sin')
    def _backward():
        value.grad += math.cos(value.data) * out.grad
    out._backward = _backward
    return out


def cos(value):
    value = Value.toValue(value)
    out = Value(math.cos(value.data), parents=(value,), op='cos')
    def _backward():
        value.grad += -math.sin(value.data) * out.grad
    out._backward = _backward
    return out


def tan(value):
    value = Value.toValue(value)
    out = Value(math.tan(value.data), parents=(value,), op='tan')
    def _backward():
        value.grad += out.grad / math.cos(value.data) ** 2
    out._backward = _backward
    return out



def arctan(value):
    value = Value.toValue(value)
    out = Value(math.atan(value.data), parents=(value,), op='arctan')
    def _backward():
        value.grad += out.grad / (1. + value.data ** 2)
    out._backward = _backward
    return out


def exp(value):
    value = Value.toValue(value)
    return value.exp()


def relu(value):
    value = Value.toValue(value)
    return value.relu()


def sqrt(value):
    value = Value.toValue(value)
    return value ** .5


def abs(value):
    value = Value.toValue(value)
    return value.abs()
    

def arcsin(value):
    # 1 / sqrt(1 - x^2)
    value = Value.toValue(value)
    out = Value(math.asin(value.data), parents=(value,), op='arcsin')
    def _backward():
        value.grad += out.grad / math.sqrt(1. - value.data ** 2)
    out._backward = _backward
    return out


def arccos(value):
    # 1 / sqrt(1 - x^2)
    value = Value.toValue(value)
    out = Value(math.acos(value.data), parents=(value,), op='arccos')
    def _backward():
        value.grad += -out.grad / math.sqrt(1. - value.data ** 2)
    out._backward = _backward
    return out


def sigmoid(value):
    value = Value.toValue(value)
    sigma = 1. / (1. + math.exp(-value.data))
    out = Value(sigma, parents=(value,), op='sigmoid')
    def _backward():
        value.grad += out.data * (1. - out.data) * out.grad
    out._backward = _backward
    return out


def tanh(value):
    value = Value.toValue(value)
    exp2 = math.exp(2 * value.data)
    th = (exp2 - 1.) / (exp2 + 1.) 
    out = Value(th, parents=(value,), op='tanh')
    def _backward():
        value.grad += (1. - out.data ** 2) * out.grad
    out._backward = _backward
    return out



def ln(value):
    value = Value.toValue(value)
    out = Value(math.log(value.data), parents=(value,), op='ln')
    def _backward():
        value.grad += out.grad / value.data
    out._backward = _backward
    return out