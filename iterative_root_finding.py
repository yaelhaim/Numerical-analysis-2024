def find_roots(f, a, b, step=0.1, tol=1e-6):
    """
    Find roots of function f within the interval [a, b]
    using a simple iterative method.
    """
    roots = []
    x = a
    while x <= b:
        if f(x) == 0:
            if not roots or abs(x - roots[-1]) > tol:
                roots.append(x)
                print(f"Root found: {x}")
        elif f(a) * f(x + step) < 0:
            x0, x1 = x, x + step
            iterations = 0
            while abs(x1 - x0) > tol:
                iterations += 1
                x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
                print(f"Iteration {iterations}: x({iterations+1}) = {x1}, f(x({iterations+1})) = {f(x1)}")
            if not roots or abs(x1 - roots[-1]) > tol:
                roots.append(x1)
                print(f"Root found: {x1}")
        x += step
    return roots




if __name__ == "__main__":
    # Define the function: x^2 - 5x + 2
    def func(x):
        """Example function: x^2 - 5x + 2."""
        return x ** 2 - 5 * x + 2


    # Find all roots within the range from 0 to 5
    roots = find_roots(func, 0, 5)
    print("\nFinal Roots :")
    for root in roots:
        print("\033[94m" + str(root) + "\033[0m")
