
def lagrange_interpolation(x_points, y_points, x):
    """
    Perform Lagrange interpolation to estimate y at point x.
    
    Args:
        x_points: List of x-coordinates of known points
        y_points: List of y-coordinates of known points
        x: Point at which to estimate y
        
    Returns:
        Estimated y value at point x
    """
    n = len(x_points)
    result = 0.0
    
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if j != i:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    
    return result

def newton_interpolation(x_points, y_points, x):
    """
    Perform Newton interpolation using divided differences.
    
    Args:
        x_points: List of x-coordinates of known points
        y_points: List of y-coordinates of known points
        x: Point at which to estimate y
        
    Returns:
        Estimated y value at point x
    """
    n = len(x_points)
    # Create divided differences table
    fdd = [[0] * n for _ in range(n)]
    
    for i in range(n):
        fdd[i][0] = y_points[i]
    
    for j in range(1, n):
        for i in range(n - j):
            fdd[i][j] = (fdd[i+1][j-1] - fdd[i][j-1]) / (x_points[i+j] - x_points[i])
    
    # Evaluate the interpolation polynomial at x
    result = fdd[0][0]
    x_term = 1.0
    
    for i in range(1, n):
        x_term *= (x - x_points[i-1])
        result += fdd[0][i] * x_term
    
    return result

import numpy as np

def cubic_spline(x_points, y_points, x):
    """
    Perform cubic spline interpolation (natural spline).
    
    Args:
        x_points: List of x-coordinates of known points (must be strictly increasing)
        y_points: List of y-coordinates of known points
        x: Point or array of points at which to estimate y
        
    Returns:
        Estimated y value(s) at point(s) x
    """
    n = len(x_points)
    h = np.diff(x_points)
    
    # Build tridiagonal system for second derivatives
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[-1, -1] = 1
    
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
    
    # Build right-hand side
    b = np.zeros(n)
    for i in range(1, n-1):
        b[i] = 3 * ((y_points[i+1] - y_points[i]) / h[i] - (y_points[i] - y_points[i-1]) / h[i-1])
    
    # Solve for second derivatives
    c_prime = np.linalg.solve(A, b)
    
    # Find which interval each x is in
    if isinstance(x, (list, np.ndarray)):
        results = []
        for x_val in x:
            i = np.searchsorted(x_points, x_val) - 1
            i = max(min(i, n-2), 0)
            dx = x_val - x_points[i]
            a = y_points[i]
            b = (y_points[i+1] - y_points[i]) / h[i] - h[i] * (2 * c_prime[i] + c_prime[i+1]) / 3
            c = c_prime[i]
            d = (c_prime[i+1] - c_prime[i]) / (3 * h[i])
            results.append(a + b * dx + c * dx**2 + d * dx**3)
        return np.array(results)
    else:
        i = np.searchsorted(x_points, x) - 1
        i = max(min(i, n-2), 0)
        dx = x - x_points[i]
        a = y_points[i]
        b = (y_points[i+1] - y_points[i]) / h[i] - h[i] * (2 * c_prime[i] + c_prime[i+1]) / 3
        c = c_prime[i]
        d = (c_prime[i+1] - c_prime[i]) / (3 * h[i])
        return a + b * dx + c * dx**2 + d * dx**3
    
def linear_regression(x_points, y_points):
    """
    Perform simple linear regression (y = mx + b).
    
    Args:
        x_points: List of x-coordinates of known points
        y_points: List of y-coordinates of known points
        
    Returns:
        Tuple of (slope, intercept) and a prediction function
    """
    n = len(x_points)
    x_mean = sum(x_points) / n
    y_mean = sum(y_points) / n
    
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_points, y_points))
    denominator = sum((x - x_mean) ** 2 for x in x_points)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    def predict(x):
        return slope * x + intercept
    
    return (slope, intercept), predict

# if __name__ == '__main__' :
#     # Example data points
#     x_data = [0, 1, 2, 3, 4]
#     y_data = [1, 3, 7, 13, 21]

#     # Test point
#     x_test = 2.5

#     # Lagrange interpolation
#     print("Lagrange:", lagrange_interpolation(x_data, y_data, x_test))

#     # Newton interpolation
#     print("Newton:", newton_interpolation(x_data, y_data, x_test))

#     # Cubic spline
#     print("Spline:", cubic_spline(x_data, y_data, x_test))

#     # Linear regression
#     params, predict = linear_regression(x_data, y_data)
#     print("Regression slope:", params[0])
#     print("Regression intercept:", params[1])
#     print("Regression prediction:", predict(x_test))