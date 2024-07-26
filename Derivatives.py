import numpy as np
import sympy as sym
from IPython.display import display

# Create symbolic variables in sympy
x = sym.symbols('x')

# Create two functions
fx = 2 * x**2
gx = 4 * x**3 - 3 * x**4


# Compute their individual derivatives
df = sym.diff(fx)
dg = sym.diff(gx)

# Display the derivatives
sym.pprint(df)
sym.pprint(dg)

# display(df) #to display nicely formatted in Jupyter NoteBook
# display(dg)
sym.pprint(sym.diff(fx*gx))