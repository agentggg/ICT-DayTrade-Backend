from django.core.management.base import BaseCommand
from ...models import Flashcard


class Command(BaseCommand):
    help = "Load 100 demo Python flashcards into the Flashcard table."

    def handle(self, *args, **options):
        flashcards = [
            # 1
            {
                "course": "python",
                "question": "What is Python?",
                "answer": "Python is a high-level, interpreted programming language.",
                "reasoning": "Python emphasizes readability and productivity, with a large standard library and support for multiple paradigms like procedural, object-oriented, and functional programming."
            },
            # 2
            {
                "course": "python",
                "question": "What is PEP 8?",
                "answer": "PEP 8 is the style guide for Python code.",
                "reasoning": "PEP 8 defines conventions for formatting Python code, such as indentation, naming, imports, and line length, to keep code consistent and readable."
            },
            # 3
            {
                "course": "python",
                "question": "How do you create a virtual environment in Python 3?",
                "answer": "Use `python -m venv env_name`.",
                "reasoning": "The `venv` module creates an isolated environment so dependencies for one project don’t affect others."
            },
            # 4
            {
                "course": "python",
                "question": "What is a virtual environment?",
                "answer": "A virtual environment is an isolated Python environment with its own interpreter and installed packages.",
                "reasoning": "Virtual environments allow different projects to use different dependency versions without conflicts."
            },
            # 5
            {
                "course": "python",
                "question": "What is the difference between a list and a tuple in Python?",
                "answer": "Lists are mutable; tuples are immutable.",
                "reasoning": "You can change, append, or remove elements in a list, but tuples cannot be modified after creation, which makes them hashable if they contain only immutable items."
            },
            # 6
            {
                "course": "python",
                "question": "How do you create a list in Python?",
                "answer": "Use square brackets, e.g., `my_list = [1, 2, 3]`.",
                "reasoning": "Lists use `[]` syntax and can hold multiple items, including mixed types, in an ordered collection."
            },
            # 7
            {
                "course": "python",
                "question": "How do you create a dictionary in Python?",
                "answer": "Use curly braces with key–value pairs, e.g., `user = {'name': 'Alice', 'age': 30}`.",
                "reasoning": "Dictionaries map keys to values and are useful for structured data lookup by key."
            },
            # 8
            {
                "course": "python",
                "question": "What is a function in Python?",
                "answer": "A function is a reusable block of code defined with `def`.",
                "reasoning": "Functions let you encapsulate logic, reduce repetition, and improve organization by grouping related operations."
            },
            # 9
            {
                "course": "python",
                "question": "How do you define a function in Python?",
                "answer": "Use the `def` keyword followed by the function name and parameters.",
                "reasoning": "For example, `def add(a, b): return a + b` creates a function that can be called multiple times with different arguments."
            },
            # 10
            {
                "course": "python",
                "question": "What is a docstring?",
                "answer": "A docstring is a string literal that documents a module, class, function, or method.",
                "reasoning": "Docstrings are written as the first statement in a definition and can be accessed via `.__doc__` or tools like `help()`."
            },
            # 11
            {
                "course": "python",
                "question": "What is a module in Python?",
                "answer": "A module is a file containing Python code that can be imported.",
                "reasoning": "Modules help organize code into reusable units; any `.py` file can act as a module."
            },
            # 12
            {
                "course": "python",
                "question": "What is a package in Python?",
                "answer": "A package is a collection of modules in a directory with an `__init__.py` file (in older versions) or treated as a namespace package.",
                "reasoning": "Packages provide a hierarchical structure to organize related modules under a common name."
            },
            # 13
            {
                "course": "python",
                "question": "How do you import a module in Python?",
                "answer": "Use the `import` statement, e.g., `import math`.",
                "reasoning": "The `import` statement loads a module so you can use its functions, classes, and variables like `math.sqrt()`."
            },
            # 14
            {
                "course": "python",
                "question": "What is the difference between `import module` and `from module import name`?",
                "answer": "`import module` imports the whole module; `from module import name` imports specific objects.",
                "reasoning": "Using `from` lets you access names directly, while `import module` keeps the module namespace explicit and avoids name clashes."
            },
            # 15
            {
                "course": "python",
                "question": "What is a list comprehension?",
                "answer": "A concise way to create lists using an expression and a loop in brackets.",
                "reasoning": "For example, `[x * 2 for x in range(5)]` is shorter and often clearer than an equivalent `for` loop plus `append`."
            },
            # 16
            {
                "course": "python",
                "question": "How do you write a basic list comprehension?",
                "answer": "Use `[expression for item in iterable]`.",
                "reasoning": "The syntax evaluates the expression for each item, returning a new list with the results."
            },
            # 17
            {
                "course": "python",
                "question": "What is a generator in Python?",
                "answer": "A generator is an iterator defined with `yield` that produces values lazily.",
                "reasoning": "Generators don’t store all values in memory; they generate them on demand, which is efficient for large data."
            },
            # 18
            {
                "course": "python",
                "question": "How do you create a generator function?",
                "answer": "Define a function with `def` that uses the `yield` keyword instead of `return`.",
                "reasoning": "Each `yield` pauses the function and returns a value, resuming from that point when the generator is iterated again."
            },
            # 19
            {
                "course": "python",
                "question": "What is the difference between `yield` and `return`?",
                "answer": "`return` exits a function and gives a single value; `yield` produces a value and pauses the function for later continuation.",
                "reasoning": "Using `yield` turns a function into a generator, enabling lazy iteration instead of returning a full collection."
            },
            # 20
            {
                "course": "python",
                "question": "What is a lambda function in Python?",
                "answer": "A lambda is an anonymous, small function defined with the `lambda` keyword.",
                "reasoning": "Lambdas are often used for short callbacks or inline operations, like sorting with a key."
            },
            # 21
            {
                "course": "python",
                "question": "How do you write a lambda that adds two numbers?",
                "answer": "`add = lambda a, b: a + b`",
                "reasoning": "Lambdas use the syntax `lambda args: expression` and automatically return the expression."
            },
            # 22
            {
                "course": "python",
                "question": "What is the `if __name__ == '__main__':` block used for?",
                "answer": "It ensures code only runs when the script is executed directly, not when imported.",
                "reasoning": "This pattern allows a file to act as both a reusable module and a standalone script."
            },
            # 23
            {
                "course": "python",
                "question": "What is the difference between `==` and `is` in Python?",
                "answer": "`==` checks value equality; `is` checks object identity.",
                "reasoning": "Two separate objects can be equal but not be the same object in memory, so `==` and `is` can behave differently."
            },
            # 24
            {
                "course": "python",
                "question": "What is a Python class?",
                "answer": "A class is a blueprint for creating objects with attributes and methods.",
                "reasoning": "Classes support object-oriented programming, letting you model real-world entities and encapsulate behavior."
            },
            # 25
            {
                "course": "python",
                "question": "How do you define a class in Python?",
                "answer": "Use the `class` keyword followed by the class name and a colon.",
                "reasoning": "For example, `class Person:` defines a class; methods inside take `self` as the first parameter."
            },
            # 26
            {
                "course": "python",
                "question": "What is `self` in a Python class?",
                "answer": "`self` is the reference to the instance the method is being called on.",
                "reasoning": "It gives methods access to instance attributes and other methods, similar to `this` in other languages."
            },
            # 27
            {
                "course": "python",
                "question": "What is inheritance in Python?",
                "answer": "Inheritance allows a class to derive from another class and reuse its behavior.",
                "reasoning": "By inheriting, a subclass can extend or override methods from the base class while sharing common functionality."
            },
            # 28
            {
                "course": "python",
                "question": "How do you create a subclass in Python?",
                "answer": "Specify the parent class in parentheses after the class name.",
                "reasoning": "For example, `class Dog(Animal):` makes `Dog` inherit attributes and methods from `Animal`."
            },
            # 29
            {
                "course": "python",
                "question": "What is method overriding?",
                "answer": "Method overriding is redefining a method in a subclass with the same name as in the parent class.",
                "reasoning": "It lets a subclass customize or replace inherited behavior for specific needs."
            },
            # 30
            {
                "course": "python",
                "question": "What does `super()` do in Python?",
                "answer": "`super()` gives access to methods of a parent or sibling class.",
                "reasoning": "It is commonly used to call the parent class’s implementation inside an overridden method, such as `super().__init__()`."
            },
            # 31
            {
                "course": "python",
                "question": "What is exception handling?",
                "answer": "Exception handling is the process of catching and managing runtime errors.",
                "reasoning": "Using `try`, `except`, `else`, and `finally` prevents crashes and lets you respond to errors gracefully."
            },
            # 32
            {
                "course": "python",
                "question": "How do you catch exceptions in Python?",
                "answer": "Use a `try` block followed by one or more `except` blocks.",
                "reasoning": "Code that may fail goes in `try`, and each `except` handles specific exception types or a general exception."
            },
            # 33
            {
                "course": "python",
                "question": "What is a `try/except/else/finally` structure?",
                "answer": "`try` contains risky code, `except` handles errors, `else` runs if no error occurs, and `finally` always runs.",
                "reasoning": "This structure lets you clearly separate success handling, error handling, and cleanup actions."
            },
            # 34
            {
                "course": "python",
                "question": "How do you raise an exception manually?",
                "answer": "Use the `raise` keyword with an exception type, e.g., `raise ValueError('message')`.",
                "reasoning": "Raising exceptions lets you signal invalid states or arguments from your own code."
            },
            # 35
            {
                "course": "python",
                "question": "What is a context manager in Python?",
                "answer": "A context manager is an object that defines `__enter__` and `__exit__` to manage resources.",
                "reasoning": "Context managers are commonly used with `with` statements to ensure resources like files are cleaned up correctly."
            },
            # 36
            {
                "course": "python",
                "question": "How do you use a context manager to open a file?",
                "answer": "Use `with open('file.txt') as f:`.",
                "reasoning": "The `with` block ensures the file is automatically closed, even if an exception occurs."
            },
            # 37
            {
                "course": "python",
                "question": "What is the difference between `list.append()` and `list.extend()`?",
                "answer": "`append()` adds a single element; `extend()` adds all elements from an iterable.",
                "reasoning": "Using `append()` with a list nests it, while `extend()` merges items into the existing list."
            },
            # 38
            {
                "course": "python",
                "question": "How do you remove an item from a list by value?",
                "answer": "Use `list.remove(value)`.",
                "reasoning": "`remove()` deletes the first matching item; it raises a `ValueError` if the value is not found."
            },
            # 39
            {
                "course": "python",
                "question": "How do you remove an item from a list by index?",
                "answer": "Use `del list[index]` or `list.pop(index)`.",
                "reasoning": "`pop()` also returns the removed item, while `del` just deletes it."
            },
            # 40
            {
                "course": "python",
                "question": "What is slicing in Python?",
                "answer": "Slicing is selecting a subsequence from sequences like lists and strings using `[start:stop:step]`.",
                "reasoning": "Slicing provides a concise way to access ranges of elements without loops."
            },
            # 41
            {
                "course": "python",
                "question": "How do you reverse a list using slicing?",
                "answer": "Use `my_list[::-1]`.",
                "reasoning": "A negative step traverses the list from end to start, returning a reversed copy."
            },
            # 42
            {
                "course": "python",
                "question": "What is a set in Python?",
                "answer": "A set is an unordered collection of unique elements.",
                "reasoning": "Sets are useful for membership tests, removing duplicates, and performing mathematical operations like union and intersection."
            },
            # 43
            {
                "course": "python",
                "question": "How do you create a set in Python?",
                "answer": "Use curly braces with values, e.g., `{1, 2, 3}`, or `set(iterable)`.",
                "reasoning": "Sets automatically remove duplicate values, keeping only unique elements."
            },
            # 44
            {
                "course": "python",
                "question": "What is dictionary comprehension?",
                "answer": "A concise way to build dictionaries using `{key: value for ...}` syntax.",
                "reasoning": "Dictionary comprehensions allow you to transform or filter data when building dictionaries in one expression."
            },
            # 45
            {
                "course": "python",
                "question": "How do you iterate over both keys and values of a dictionary?",
                "answer": "Use `for key, value in my_dict.items():`.",
                "reasoning": "The `.items()` method returns key–value pairs, making iteration more convenient."
            },
            # 46
            {
                "course": "python",
                "question": "What is the `in` operator used for in Python?",
                "answer": "It checks membership in sequences, sets, and dictionaries.",
                "reasoning": "Using `in` is a simple, readable way to see if a value exists in a container, like `x in my_list`."
            },
            # 47
            {
                "course": "python",
                "question": "What does `len()` do?",
                "answer": "`len()` returns the number of items in a container.",
                "reasoning": "It works with strings, lists, tuples, sets, dictionaries, and many custom objects that implement `__len__`."
            },
            # 48
            {
                "course": "python",
                "question": "What is the `range()` function used for?",
                "answer": "`range()` generates a sequence of integers.",
                "reasoning": "It is commonly used for `for` loops, e.g., `for i in range(5):` iterates over 0 through 4."
            },
            # 49
            {
                "course": "python",
                "question": "How do you iterate over a list with indexes in Python?",
                "answer": "Use `enumerate(my_list)`.",
                "reasoning": "`enumerate()` yields `(index, value)` pairs, making index-based iteration concise and clear."
            },
            # 50
            {
                "course": "python",
                "question": "What does `zip()` do in Python?",
                "answer": "`zip()` combines multiple iterables into tuples of corresponding elements.",
                "reasoning": "It is useful when you want to iterate over multiple sequences in parallel, like `for a, b in zip(list1, list2):`."
            },
            # 51
            {
                "course": "python",
                "question": "What is `*args` in a function definition?",
                "answer": "`*args` collects extra positional arguments into a tuple.",
                "reasoning": "It allows functions to accept a variable number of arguments without specifying them individually."
            },
            # 52
            {
                "course": "python",
                "question": "What is `**kwargs` in a function definition?",
                "answer": "`**kwargs` collects extra keyword arguments into a dictionary.",
                "reasoning": "It lets functions accept dynamic, named arguments that you can access by key."
            },
            # 53
            {
                "course": "python",
                "question": "What are default parameter values in Python functions?",
                "answer": "They are preset values used when a caller does not supply that argument.",
                "reasoning": "Defining `def greet(name='World'):` lets you call `greet()` or `greet('Alice')` flexibly."
            },
            # 54
            {
                "course": "python",
                "question": "Why should you avoid using mutable objects as default parameter values?",
                "answer": "Because the same object is reused across calls, causing unexpected shared state.",
                "reasoning": "Defaults are evaluated once at function definition time, so using lists or dicts can lead to subtle bugs."
            },
            # 55
            {
                "course": "python",
                "question": "What is a decorator in Python?",
                "answer": "A decorator is a function that takes another function and extends or modifies its behavior.",
                "reasoning": "Decorators use `@` syntax and are useful for cross-cutting concerns like logging, authentication, or timing."
            },
            # 56
            {
                "course": "python",
                "question": "How do you write a simple decorator?",
                "answer": "Define a wrapper function inside another function and return the wrapper.",
                "reasoning": "The wrapper can run code before and after calling the original function, allowing behavior injection."
            },
            # 57
            {
                "course": "python",
                "question": "What is the difference between `print()` and `return`?",
                "answer": "`print()` displays output to the console; `return` sends a value back to the caller.",
                "reasoning": "Functions should generally use `return` to provide results; printing is side-effect output for humans."
            },
            # 58
            {
                "course": "python",
                "question": "What type does the `input()` function return?",
                "answer": "`input()` always returns a string.",
                "reasoning": "If you need numeric values, you must convert the result, e.g., with `int()` or `float()`."
            },
            # 59
            {
                "course": "python",
                "question": "What is `__init__.py` used for in a package?",
                "answer": "It marks a directory as a Python package (in older versions) and can run initialization code.",
                "reasoning": "It also controls what gets imported with `from package import *` via the `__all__` list."
            },
            # 60
            {
                "course": "python",
                "question": "What is the Global Interpreter Lock (GIL)?",
                "answer": "The GIL is a mutex that allows only one thread to execute Python bytecode at a time.",
                "reasoning": "It simplifies memory management but can limit CPU-bound multi-threaded performance in CPython."
            },
            # 61
            {
                "course": "python",
                "question": "When are lists preferred over tuples?",
                "answer": "Lists are preferred when you need a mutable sequence.",
                "reasoning": "You can add, remove, or change items in a list, which is not allowed with tuples."
            },
            # 62
            {
                "course": "python",
                "question": "When are tuples preferred over lists?",
                "answer": "Tuples are preferred when you need an immutable, fixed collection or want to use it as a dictionary key.",
                "reasoning": "Immutability can provide safety and makes tuples hashable if they contain only immutable items."
            },
            # 63
            {
                "course": "python",
                "question": "What is list unpacking in Python?",
                "answer": "List unpacking assigns elements of a list (or iterable) to multiple variables in one statement.",
                "reasoning": "For example, `a, b, c = [1, 2, 3]` is more expressive than indexing individually."
            },
            # 64
            {
                "course": "python",
                "question": "How do you ignore values when unpacking?",
                "answer": "Use `_` or `*rest` for values you don’t care about.",
                "reasoning": "For example, `a, _, c = (1, 2, 3)` signals that the middle value is intentionally unused."
            },
            # 65
            {
                "course": "python",
                "question": "What is a frozen set?",
                "answer": "A frozenset is an immutable version of a set.",
                "reasoning": "Since it cannot be changed, it can be used as a key in dictionaries or stored in other sets."
            },
            # 66
            {
                "course": "python",
                "question": "What is the `any()` function used for?",
                "answer": "`any()` returns True if any element in an iterable is truthy.",
                "reasoning": "It short-circuits on the first True value, making it efficient for checks."
            },
            # 67
            {
                "course": "python",
                "question": "What is the `all()` function used for?",
                "answer": "`all()` returns True only if all elements in an iterable are truthy.",
                "reasoning": "It short-circuits on the first False value, and is commonly used for validation checks."
            },
            # 68
            {
                "course": "python",
                "question": "What does `sorted()` do?",
                "answer": "`sorted()` returns a new sorted list from any iterable.",
                "reasoning": "It does not modify the original iterable and accepts a `key` and `reverse` argument for custom sorting."
            },
            # 69
            {
                "course": "python",
                "question": "What is the difference between `sorted()` and `list.sort()`?",
                "answer": "`sorted()` returns a new list; `list.sort()` sorts the list in place and returns None.",
                "reasoning": "Use `sorted()` when you need a new sorted result, and `list.sort()` when you want to modify the original list."
            },
            # 70
            {
                "course": "python",
                "question": "How do you read all lines from a file into a list?",
                "answer": "Use `f.readlines()` inside a `with open(...) as f:` block.",
                "reasoning": "`readlines()` returns a list where each element is a line from the file, including newline characters."
            },
            # 71
            {
                "course": "python",
                "question": "How do you iterate over a file line by line?",
                "answer": "Iterate directly over the file object, e.g., `for line in f:`.",
                "reasoning": "File objects are iterable, and this approach is memory-efficient for large files."
            },
            # 72
            {
                "course": "python",
                "question": "What does `strip()` do on a string?",
                "answer": "`strip()` removes leading and trailing whitespace by default.",
                "reasoning": "It’s commonly used to clean input or lines read from files."
            },
            # 73
            {
                "course": "python",
                "question": "How do you split a string into a list of words?",
                "answer": "Use `s.split()`.",
                "reasoning": "By default, `split()` separates the string on any whitespace and returns a list of substrings."
            },
            # 74
            {
                "course": "python",
                "question": "How do you join a list of strings into one string with commas?",
                "answer": "Use `','.join(list_of_strings)`.",
                "reasoning": "`join()` is called on the separator string and combines all elements of the iterable, inserting the separator between them."
            },
            # 75
            {
                "course": "python",
                "question": "What is f-string formatting in Python?",
                "answer": "F-strings are string literals prefixed with `f` that allow inline expression interpolation.",
                "reasoning": "For example, `name = 'Alice'; f'Hello, {name}'` is concise and readable compared to older formatting methods."
            },
            # 76
            {
                "course": "python",
                "question": "How do you format a number to two decimal places using an f-string?",
                "answer": "Use `f'{value:.2f}'`.",
                "reasoning": "The `:.2f` format specifier rounds the number to two decimal places as a string."
            },
            # 77
            {
                "course": "python",
                "question": "What is type hinting in Python?",
                "answer": "Type hinting adds optional type information to variables, function parameters, and return values.",
                "reasoning": "Hints help tools and developers understand expected types, improving readability and static analysis."
            },
            # 78
            {
                "course": "python",
                "question": "How do you annotate the return type of a function in Python?",
                "answer": "Use `->` followed by the type after the parameter list.",
                "reasoning": "For example, `def add(a: int, b: int) -> int:` documents that the function returns an integer."
            },
            # 79
            {
                "course": "python",
                "question": "What is the `typing` module used for?",
                "answer": "The `typing` module provides type hints and generic types for static type checking.",
                "reasoning": "It includes types like `List`, `Dict`, `Optional`, and `Union` that describe complex structures."
            },
            # 80
            {
                "course": "python",
                "question": "What does `pass` do in Python?",
                "answer": "`pass` is a no-op statement that does nothing.",
                "reasoning": "It’s useful as a placeholder in blocks where code is syntactically required but not yet implemented."
            },
            # 81
            {
                "course": "python",
                "question": "What is the `None` object in Python?",
                "answer": "`None` represents the absence of a value.",
                "reasoning": "It is often used for default arguments, missing data, or explicit 'no result' returns."
            },
            # 82
            {
                "course": "python",
                "question": "What is the difference between `break` and `continue` in loops?",
                "answer": "`break` exits the loop entirely; `continue` skips to the next iteration.",
                "reasoning": "They provide control flow tools to handle special conditions inside loops."
            },
            # 83
            {
                "course": "python",
                "question": "What does the `else` clause on a loop do in Python?",
                "answer": "The loop `else` runs if the loop completes normally without hitting a `break`.",
                "reasoning": "It is useful to detect if a `for` or `while` loop terminated naturally instead of early exit."
            },
            # 84
            {
                "course": "python",
                "question": "What is a `while` loop used for?",
                "answer": "A `while` loop repeatedly executes as long as a condition is true.",
                "reasoning": "It is helpful when the number of iterations is not known in advance and depends on runtime conditions."
            },
            # 85
            {
                "course": "python",
                "question": "What is a `for` loop used for in Python?",
                "answer": "`for` loops iterate over items of a sequence or any iterable.",
                "reasoning": "Python’s `for` loop abstracts away index management and works directly with iterables."
            },
            # 86
            {
                "course": "python",
                "question": "What is the difference between a shallow copy and a deep copy?",
                "answer": "A shallow copy copies the top-level container; a deep copy copies the container and all nested objects.",
                "reasoning": "Shallow copies share nested objects, while deep copies recursively duplicate them, avoiding shared references."
            },
            # 87
            {
                "course": "python",
                "question": "How do you make a shallow copy of a list?",
                "answer": "Use `list.copy()`, `list[:]`, or `list(list_obj)`.",
                "reasoning": "All of these create a new list object that shares references to the same elements."
            },
            # 88
            {
                "course": "python",
                "question": "How do you make a deep copy of an object?",
                "answer": "Use `copy.deepcopy(obj)` from the `copy` module.",
                "reasoning": "Deep copy recursively duplicates nested objects, creating a fully independent structure."
            },
            # 89
            {
                "course": "python",
                "question": "What is `__repr__` used for?",
                "answer": "`__repr__` returns an unambiguous string representation of an object, mainly for debugging.",
                "reasoning": "It should, when possible, look like valid Python code that could recreate the object."
            },
            # 90
            {
                "course": "python",
                "question": "What is `__str__` used for?",
                "answer": "`__str__` returns a readable, user-friendly string representation of an object.",
                "reasoning": "It is used by `print()` and `str()` to show objects in a human-oriented way."
            },
            # 91
            {
                "course": "python",
                "question": "How do you check the type of a variable in Python?",
                "answer": "Use the built-in `type()` function.",
                "reasoning": "`type(x)` returns the object's type, which is useful for debugging and conditional logic."
            },
            # 92
            {
                "course": "python",
                "question": "What does `isinstance()` do?",
                "answer": "`isinstance(obj, cls)` checks if an object is an instance of a class or its subclasses.",
                "reasoning": "It’s preferred over direct type checks because it supports inheritance."
            },
            # 93
            {
                "course": "python",
                "question": "What is an iterator in Python?",
                "answer": "An iterator is an object that implements `__iter__()` and `__next__()` to return items one at a time.",
                "reasoning": "Iterators provide a unified way to loop over different container types lazily."
            },
            # 94
            {
                "course": "python",
                "question": "What is an iterable in Python?",
                "answer": "An iterable is any object that can return an iterator, typically implementing `__iter__()`.",
                "reasoning": "Iterables can be used in `for` loops and other contexts that consume sequences of items."
            },
            # 95
            {
                "course": "python",
                "question": "What is the purpose of the `with` statement?",
                "answer": "`with` ensures resources are properly acquired and released by using context managers.",
                "reasoning": "It simplifies resource management like opening files, acquiring locks, or managing transactions."
            },
            # 96
            {
                "course": "python",
                "question": "What is a namedtuple?",
                "answer": "A namedtuple is a tuple subclass with named fields, from the `collections` module.",
                "reasoning": "It lets you access elements by name as well as by index, improving code clarity."
            },
            # 97
            {
                "course": "python",
                "question": "What is `collections.Counter` used for?",
                "answer": "`Counter` counts hashable objects and stores frequencies in a dictionary-like structure.",
                "reasoning": "It’s useful for tasks like counting words, characters, or events."
            },
            # 98
            {
                "course": "python",
                "question": "What is `itertools` in Python?",
                "answer": "`itertools` is a module providing fast, memory-efficient tools for working with iterators.",
                "reasoning": "It includes functions like `cycle`, `chain`, `product`, and `permutations` for advanced iteration patterns."
            },
            # 99
            {
                "course": "python",
                "question": "What is `enumerate()` useful for?",
                "answer": "`enumerate()` adds a counter to an iterable and returns index–value pairs.",
                "reasoning": "It simplifies loops where you need both the index and the item without manually managing a counter."
            },
            # 100
            {
                "course": "python",
                "question": "What command do you use to install a package with pip?",
                "answer": "Use `pip install package_name`.",
                "reasoning": "`pip` is Python’s package installer, and this command downloads and installs the package from PyPI or another index."
            },
        ]

        created_count = 0
        for card in flashcards:
            obj, created = Flashcard.objects.get_or_create(
                question=card["question"],
                course=card["course"],
                defaults={
                    "answer": card["answer"],
                    "reasoning": card["reasoning"],
                },
            )
            if created:
                created_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Inserted {created_count} new Python flashcards (out of {len(flashcards)})."
            )
        )