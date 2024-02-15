#  MERGRAD

**Mergrad** is a lightweight library for automatic differentiation.


### Usage

The basic unit is an object of class Value.
Any object of this class allows to build a calculation tree and then read the gradient from it. 


```
x = Value(2.)
y = x ** 2
```

When using objects of this class in arithmetic calculations, the calculation tree will be automatically built.

![some text](https://github.com/gitmskhl/mergrad/blob/main/images/im1.png)


To calculate the gradients, it is enough to call the .backward() method on the variable that saved the result of the calculation.

```
x = Value(2.)
y = x ** 2
y.backward()
```
The calculation tree looks like this

![some text](https://github.com/gitmskhl/mergrad/blob/main/images/im2.png)


The gradient values after the .backward() method is called are stored in the .grad variable field.

```
from mergrad.functional import sin

x = Value(1.6)
y = Value(2.5)
z = sin(x ** 2 + x * y)
z.backward()
print(x.grad)
print(y.grad)
```

Output:

```
5.483005781936319
1.5390893422979142
```

![some text](https://github.com/gitmskhl/mergrad/blob/main/images/im3.png)
