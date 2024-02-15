import torch
from mergrad.value import Value
from mergrad.functional import *


def _abs(a):
    return a if a >= 0 else -a


def isclose(a, b, tol=1e-1):
    return _abs(a.item() - b) < tol


def ex1(x_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = torch.sin(3 * x - 5)
    y.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = sin(3 * x - 5)
    y.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)


def ex2(x_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = (2 * x + 1) ** 5
    y.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = (2 * x + 1) ** 5
    y.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)



def ex3(x_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = 1. / (x ** 2 - 1) ** 7
    y.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = 1. / (x ** 2 - 1) ** 7
    y.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)


def ex4(x_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = torch.arctan(torch.sqrt(x))
    y.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = arctan(sqrt(x))
    y.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)


def ex5(x_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = (x ** 2 + torch.tan(x) + 15) ** (1 / 3)
    y.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = (x ** 2 + tan(x) + 15) ** (1 / 3)
    y.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)



def ex6(x_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = 5 / (x + torch.log(x)) ** (1 / 5)
    y.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = 5 / (x + ln(x)) ** (1 / 5)
    y.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)



def ex7(x_, y_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([y_], dtype=torch.float32, requires_grad=True)
    z = x ** 2 * y - 4 * x * torch.sqrt(y) - 6 * y ** 2 + 5
    z.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = Value(y_)
    z = x ** 2 * y - 4 * x * sqrt(y) - 6 * y ** 2 + 5
    z.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)


def ex8(x_, y_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([y_], dtype=torch.float32, requires_grad=True)
    z = y * torch.sin(2 * y) / x ** (2 / 3)
    z.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = Value(y_)
    z = y * sin(2 * y) / x ** (2 / 3)
    z.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)


def ex9(x_, y_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([y_], dtype=torch.float32, requires_grad=True)
    z = torch.exp(x) * (torch.cos(y) + x * torch.sin(y))
    z.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = Value(y_)
    z = exp(x) * (cos(y) + x * sin(y))
    z.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)


def ex10(x_, y_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([y_], dtype=torch.float32, requires_grad=True)
    z = torch.sin(torch.sqrt(y / x ** 3))
    z.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = Value(y_)
    z = sin(sqrt(y / x ** 3))
    z.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)


def ex11(x_, y_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([y_], dtype=torch.float32, requires_grad=True)
    z = torch.arctan(x * torch.sqrt(y))
    z.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = Value(y_)
    z = arctan(x * sqrt(y))
    z.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)


def ex12(x_, y_):
    # torch
    x = torch.tensor([x_], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([y_], dtype=torch.float32, requires_grad=True)
    z = 2 ** y / y + x ** 2 * torch.tan(x) + torch.log(x ** 2 + y ** 3)
    z.backward()
    torch_grad = x.grad
    
    # mergrad
    x = Value(x_)
    y = Value(y_)
    z = 2 ** y / y + x ** 2 * tan(x) + ln(x ** 2 + y ** 3)
    z.backward()
    mer_grad = x.grad
    return isclose(torch_grad, mer_grad)



def testValue():
    
    # one argument
    print('-' * 20, 'ONE ARUMENT', '-' * 20)
    
    X = [1.2, 3., 2.6]
    # torch
    
    functs = [ex1, ex2, ex3, ex4, ex5, ex6]
    
    for i, fun in enumerate(functs):
        for x in X:
            if fun(x):
                print('Example %i x=%f status: OK' % (i + 1, x))
            else:
                print('Example %i x=%f status: FAIL' % (i + 1, x))
                return
            
    
    # multiple arguments
    
    print('\n', '-' * 20, 'MULTIPLE ARUMENTS', '-' * 20)
    
    X = [4.6, 5.8, 2]
    Y = [7.4, 9, 1.64]
    functs = [ex7, ex8, ex9, ex10, ex11, ex12]
    for i, fun in enumerate(functs):
        for x, y in zip(X, Y):
            if fun(x, y):
                print('Example %i x=%f y=%f status: OK' % (i + 7, x, y))
            else:
                print('Example %i x=%f y=%f status: FAIL' % (i + 7, x, y))
                return
            

if __name__ == "__main__:
    testValue()
