import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize

#Define math functions
def evalute_expression(expression):
    try:
        expr = parse_expr(expression)
        result = expr.evalf()
        return result
    except Exception as e:
        return str(e)

def basic_arithmetic(operation, num1, num2):
    if operation == 'add':
        return num1 + num2
    elif operation == 'subtract':
        return num1 - num2
    elif operation == 'multiply':
        return num1 * num2
    elif operation == 'divide':
        if num2!= 0:
            return num1 / num2
        else:
            return "Error: Division by zero."

def handle_input(input_str):
    if '+' in input_str:
        num1, num2 = input_str.split('+')
        return basic_arithmetic('add',float(num1),float(num2))
    elif'-' in input_str:
        num1, num2 = input_str.split('-')
        return basic_arithmetic('subtract',float(num1),float(num2))
    elif'*' in input_str:
        num1, num2 = input_str.split('*')
        return basic_arithmetic('multipy',float(num1),float(num2))
    elif'/' in input_str:
        num1, num2 = input_str.split('/')
        return basic_arithmetic('divide',float(num1),float(num2))
    else:
        return evalute_expression(input_str)

def graph_function(func,x_range):
    x = np.linspace(x_range[0], x_range[1],400)
    y = sp.lambdify('x', parse_expr(func))(x)
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Graph of {func}')
    plt.show()

def integrate_function(func, lower_limit, upper_limit):
    integral, error = quad(sp.lambdify('x', parse_expr(func)), lower_limit, upper_limit)
    return integral

def minimize_function(func, initial_guess):
    result = minimize(sp.lambdify('x', parse_expr(func)), initial_guess)
    return result.x[0]

def derivative_function(func):
    try:
        x = sp.symbols('x')
        expr = parse_expr(func)
        derivative = sp.diff(expr,x)
        return derivative
    except Exception as e:
        print(f"Error:{e}")

def cubic_function(a, b, c, d):
    try:
        x = sp.symbols('x')
        expr = a*x**3 + b*x**2 + c*x + d
        return expr
    except Exception as e:
        print(f"Error{e}")

def solve_simultaneously(eq1, eq2):
    x,y = sp.symbols('x y')
    eq1 = parse_expr(eq1)
    eq2 = parse_expr(eq2)
    solutions = sp.solve((eq1, eq2), (x,y))
    return solutions

# Define chatbot functions
def math_chatbot():
    print("Welcome to Mathbot!")
    while True:
        user_input = input("You: ")
        user_input = user_input.lower()

        if user_input in ["hi","hello","hey"]:
            print("MathBot: Hello! How can I assist you today?")

        elif user_input in ["help","commands"]:
            print("MathBot: Available commands:")
            print(" - calculate [expression]")
            print(" -graph [function][x-range]")
            print(" - integrate [function][lower limit][upper limit]")
            print(" - minimize [function][initial guess]")
            print(" - solve [equation 1][equation 2]")
            print(" - quit")

        elif user_input.startswith("calculate"):
            expression = user_input.split(" ")[1]
            result = handle_input(expression)
            print(f"MathBot: Result: {result}")

        elif user_input.startswith("graph "):
            func, x_range = user_input.split(" ")[1],[float(x) for x in user_input.split(" ")[2:4]]
            graph_function(func,x_range)
            print("MathBot: Graph displayed.")

        elif user_input.startswith("integrate "):
            func, lower_limit, upper_limit = user_input.split(" ")[1], float(user_input.split(" ")[2]), float(user_input.split(" ")[3])
            result = integrate_function(func, lower_limit, upper_limit)
            print(f"MathBot: Integral: {result}")

        elif user_input.startswith("minimize "):
            func, initial_guess = user_input.split(" ")[1], float(user_input.split(" ")[2])
            result = minimize_function(func, initial_guess)
            print(f"MathBot: Minimum: {result}")

        elif user_input.startswith("solve"):
            eq1, eq2 = user_input.split(" ")[2]
            solutions = solve_simultaneously(eq1, eq2)
            print(f"MathBot: Solutions: {solutions}")

        elif user_input == "quit":
            print("MathBot: Goodbye!")
            break

        else:
            print("MathBot: Invalid command. Type 'help' for available commands.")

#Run chatbot
math_chatbot()
