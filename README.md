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


