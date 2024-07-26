import numpy as np
import matplotlib.pyplot as plt
from IPython import display 
display.set_matplotlib_formats("svg")
def fx(x):
    return 3*x**2 - 3*x + 4
def deriv(x):
    return 6*x-3
x = np.linspace(-2, 2, 2001)
plt.plot(x, fx(x), x,deriv(x))
plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(["y","dy"])
plt.show()
localmin = np.random.choice(x,1)
learning_rate = .01
training_epochs = 100

for i in range(training_epochs):
    grad=deriv(localmin)
    localmin = localmin - learning_rate*grad
print(localmin)
plt.plot(x,fx(x), x,deriv(x))
plt.plot(localmin, deriv (localmin), 'ro')
plt.plot(localmin,fx(localmin), 'ro')
plt.xlim(x[[0,-1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)', 'df','f(x) min'])
plt.title('Empirical local minimum: %s'%localmin[0])
plt.show()