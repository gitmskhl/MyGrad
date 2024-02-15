import math
from numbers import Number

'''
(C) Musa Khalilov 15.02.2024
'''


class Value:
    def __init__(self, data, parents=(), op=''):
        self.data = data
        self.grad = 0.
        self.topsorted = None
        self._backward = lambda: None
        self._parents = parents
        self._op = op
    
    
    def __repr__(self):
        return "Value(data=%f)" % self.data

    
    def backward(self):
        if self.topsorted is None:
            topsorted = []
            visited = set()
            def topsort(v):
                visited.add(v)
                for p in v._parents:
                    if p not in visited: topsort(p)
                topsorted.append(v)
            topsort(self)
            
            self.grad = 1.
            for n in reversed(topsorted):
                n._backward()
        
    
    
    def toValue(val):
        if isinstance(val, Value): return val
        assert type(val) in [int, float], 'The type of element must be either int or float'
        return Value(data=val)
    

    def __add__(self, other):
        other = Value.toValue(other)
        out = Value(self.data + other.data, parents=(self, other), op='+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    
    def __radd__(self, other):
        return self + other
    
    
    def __neg__(self):
        out = Value(-self.data, parents=(self,), op='-')
        def _backward():
            self.grad += -out.grad
            
        out._backward = _backward
        return out
    
    
    def __sub__(self, other):
        other = Value.toValue(other)
        out = Value(self.data - other.data, parents=(self, other), op='-')
        def _backward():
            self.grad += out.grad
            other.grad += -out.grad
        out._backward = _backward
        return out
    
    
    def __rsub__(self, other):
        return other + (-self)
    
    
    def __mul__(self, other):
        other = Value.toValue(other)
        out = Value(self.data * other.data, parents=(self, other), op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    
    def __rmul__(self, other):
        return self * other
    
    
    def __truediv__(self, other):
        other = Value.toValue(other)
        out = Value(self.data / other.data, parents=(self, other), op='/')
        def _backward():
            self.grad += out.grad / other.data
            other.grad += -(self.data * out.grad) / other.data ** 2
        out._backward = _backward
        return out
    
    # a util method
    def _inverse_(self):
        out = Value(1. / self.data, parents=(self,), op='inv')
        def _backward():
            self.grad += -out.grad / self.data ** 2
        out._backward = _backward
        return out
    
    
    def _pow_number_(self, number):
        out = Value(self.data ** number, parents=(self,), op='**')
        def _backward():
            self.grad += (number * self.data ** (number - 1)) * out.grad
        out._backward = _backward
        return out
    
    
    def __rtruediv__(self, other):
        return other * self._inverse_()
    
    
    def __pow__(self, other):
        if isinstance(other, Number): return self._pow_number_(other)
        out = Value(self.data ** other.data, parents=(self, other), op='**')
        def _backward():
            assert self.data > 0, "a^b: Can't compute the gradient in the power function if a is less than zero"
            self.grad += (out.data * other.data / self.data) * out.grad  # a^b * b / a
            other.grad += (out.data * math.log(self.data)) * out.grad # a^b * ln(a)
        out._backward = _backward
        return out
    
    
    def __rpow__(self, other):
        other = Value.toValue(other)
        return other ** self
    
    
    def exp(self):
        out = Value(math.exp(self.data), parents=(self,), op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
        
    
    def relu(self):
        out = Value(self.data if self.data > 0 else 0., parents=(self,), op='relu')
        def _backward():
            self.grad += out.grad if self.data > 0 else 0.
        out._backward = _backward
        return out
    
    
    def abs(self):
        out = Value(abs(self.data), parents=(self,), op='abs')
        def _backward():
            self.grad += out.grad if self.data > 0 else -out.grad
        out._backward = _backward
        return out

