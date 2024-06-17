import numpy as np
import tenseal as ts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sympy

# Create TenSEAL context with proper key generation
def create_context(poly_modulus_degree, plain_modulus):
    """
    Creates a TenSEAL context with specified polynomial modulus degree and plaintext modulus.
    Generates Galois keys and relinearization keys for the context.
    """
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=poly_modulus_degree,
        plain_modulus=plain_modulus
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

# Encrypt data
def encrypt_data(context, data):
    """
    Encrypts the given data using the provided TenSEAL context.
    """
    return ts.bfv_vector(context, data)

# Decrypt data
def decrypt_data(enc_data):
    """
    Decrypts the given encrypted data.
    """
    return np.array(enc_data.decrypt())

# Train a simple linear regression model on encrypted data
def train_model_on_encrypted_data(enc_X, enc_y, context):
    """
    Trains a linear regression model on encrypted data.
    Decrypts the data for training purposes.
    """
    model = LinearRegression()
    X = np.array([decrypt_data(vec) for vec in enc_X])
    y = np.array([decrypt_data(vec) for vec in enc_y]).ravel()
    model.fit(X, y)
    return model

# Evaluate the model
def evaluate_model(model, enc_X, enc_y):
    """
    Evaluates the linear regression model on encrypted data.
    """
    X = np.array([decrypt_data(vec) for vec in enc_X])
    y = np.array([decrypt_data(vec) for vec in enc_y]).ravel()
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred)

# Generate random data
def generate_random_data(num_samples, num_features, noise=0.1):
    """
    Generates random data for training.
    """
    X = np.random.rand(num_samples, num_features) * 10
    true_coefficients = np.random.rand(num_features)
    y = X.dot(true_coefficients) + np.random.randn(num_samples) * noise
    return X, y

# Generate valid plain moduli
def generate_valid_plain_moduli(poly_modulus_degree, count):
    """
    Generates a list of valid plaintext moduli for a given polynomial modulus degree.
    """
    valid_plain_moduli = []
    modulus_base = 2 * poly_modulus_degree
    candidate = modulus_base + 1
    while len(valid_plain_moduli) < count:
        if sympy.isprime(candidate):
            valid_plain_moduli.append(candidate)
        candidate += modulus_base
    return valid_plain_moduli

# Define method to generate poly modulus degrees
def generate_poly_modulus_degrees(size):
    """
    Generates a list of polynomial modulus degrees.
    """
    return [2**i for i in range(12, 12 + size)]

# Main script to perform parameter search and model evaluation
if __name__ == "__main__":
    poly_modulus_degrees_size = 6
    plain_modulus_counts = 2

    poly_modulus_degrees = generate_poly_modulus_degrees(poly_modulus_degrees_size)

    param_grid = []
    for degree in poly_modulus_degrees:
        valid_plain_moduli = generate_valid_plain_moduli(degree, plain_modulus_counts)
        for plain_modulus in valid_plain_moduli:
            param_grid.append({'poly_modulus_degree': degree, 'plain_modulus': plain_modulus})

    best_mse = float('inf')
    best_params = None

    # Generate random data
    X, y = generate_random_data(num_samples=100, num_features=2)

    # Grid search for optimal parameters
    for params in param_grid:
        try:
            context = create_context(params['poly_modulus_degree'], params['plain_modulus'])
            enc_X = [encrypt_data(context, row.tolist()) for row in X]
            enc_y = [encrypt_data(context, [val]) for val in y]
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

        model = train_model_on_encrypted_data(enc_X, enc_y, context)
        mse = evaluate_model(model, enc_X, enc_y)
        print(f"Params: {params}, MSE: {mse}")

        if mse < best_mse:
            best_mse = mse
            best_params = params

    print(f"Best Params: {best_params}, Best MSE: {best_mse}")
