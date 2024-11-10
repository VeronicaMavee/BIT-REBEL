import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

#math Expression Parser
def parse_math_Expression(expr):
    return parse_expr(expr)

#knowledge Graph 
knowledge_graph = {
    'algebra': ['equations','inequalities', 
'systems'],
     'calculus': ['limits', 'derivatives',
'integrals'],
'geometry': ['points', 'lines', 'planes'],
'trigonometry': ['triangles', 'circles', 'waves']
}

#Reasoning Engine 
def apply_math_rule(expr, rule):
    #Apply mathemetical rules and theorems
    if rule == 'solve_equation':
        return sp.solve(expr)
    elif rule == 'find_derevative':
        return sp.diff(expr)
    elif rule == 'find_integral':
        return sp.integral(expr)
    elif rule == 'find_limit':
        return sp.limit(expr)
    
#Machine learning 
def train_model(X, Y):
    model = RandomForestRegressor()
    model.fit(X, Y)
    return model
    
# NLP
def generate_math_problem(concept):
    problem = ""
    if concept == 'algebra':
        problem = "Solve for x: 2x + 3 = 5"
    elif concept == 'calculus':
        problem = "Find the derivative of f(x) = x^2"
    elif concept == 'geometry':
        problem = "Find the distance between points (1, 2) and (4, 6)"
    elif concept == 'trigonometry':
        problem = "Solve or x: sin(x) = 0.5"
    return problem

# Math Solver AI
def math_solver_ai(problem):
    expr = parse_math_expression(problem)
    concept = determine_concept(expr)
    solution = apply_math_rule(expr, concept)
    return solution

def determine_concept(expr):
    # Determine math concept based on expression
    if expr.is_Eq:
        return 'algebra'
    elif expr.is_Derivative:
        return 'calculus'
    elif expr.is_Add or expr.is_Mul:
        return 'algebra'
    elif expr.is_Function:
        return 'calculus'
    else:
        return 'unknown'
    
def plot_function(func, x_range):
    x = np.linspace(x_range[0], x_range[1, 400])
    y = sp.lambdify('x', func)(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Graph of {func}')
    plot.show()
    
# Test Math Solver AI
problem = input("Enter a math problem: ")
solution = math_solver_ai(problem)
print(f"Solution: {solution}")

# Chatbot Interface
def chatbot():
    print("Welcome to MathBot!")

    commands = {
        "help": "Display available commands",
        "solve": "Solve a math problem",
        "plot": "Plot a function",
        "generate": "Generate a math problem",
        "train": "Train a machine learrning model",
        "quit": "Exit the chatbot"
    }

    while True:
        user_input = input("You: ")
        user_input = user_input.lower()

        if user_input in ["hi", "hello", "hey"]:
            print("MathBot: Hello! How can I assist you today?")
        elif user_input == "help":
            print("MathBot: Available commands: ")
            for command, description in commands.items():
                print(f"- {command}: {description}")
        elif user_input.startswith("solve "):
            problem = user_input.split(" ")[1]
            solution = math_solver_ai(problem)
            print(f"MathBot: Ssolution: {solution}")
        elif user_input.startswith("plot "):
            func, x_range = user_input.split(" ")[1], [float(x) for x in user_input.split(" ")[2:4]]
            plot_function(parse_expr(func), x_range)
            print("MathBot: Graph displayed. ")
        elif user_input.startswith("generate"):
            concept = user_input.split(" ")[1]
            problem = generate_math_problem(concept)
            print(f"MathBot: Problem: {problem}")
        elif user_input.startswith("train"):
            X = np.array([1, 2, 3, 4, 5])
            y = np.array([2, 4, 6, 8, 10])
            model = train_model(X, y)
            print(f"MathBot: Model Trained: {model}")
        elif user_input == "quit":
            print("MathBot: Goodbye!")
            break
        else:
            print("MathBot: Invalid command. Type 'help' for available commands.")

chatbot()