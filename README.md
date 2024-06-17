## README

### Machine Learning for Optimization of Homomorphic Encryption

This project aims to use machine learning techniques to optimize the parameters of homomorphic encryption schemes, specifically targeting the optimization of the BFV scheme in the TenSEAL library. The primary goal is to train a linear regression model on encrypted data and find the best parameters for the encryption scheme that minimize the mean squared error (MSE) of the model.

### Files
- `homomorphic encryption optimization.py`: The main Python script containing all necessary functions and the main execution logic.

### Prerequisites
Before running the script, ensure you have the following dependencies installed:
- `numpy`
- `scikit-learn`
- `tenseal`
- `sympy`

You can install the required packages using pip:
```bash
pip install numpy scikit-learn tenseal sympy
```

### Script Overview
The script includes several functions and the main execution block:

#### Functions
1. **create_context(poly_modulus_degree, plain_modulus)**:
   - Creates a TenSEAL context with specified polynomial modulus degree and plaintext modulus.
   - Generates Galois keys and relinearization keys for the context.

2. **encrypt_data(context, data)**:
   - Encrypts the given data using the provided TenSEAL context.

3. **decrypt_data(enc_data)**:
   - Decrypts the given encrypted data.

4. **train_model_on_encrypted_data(enc_X, enc_y, context)**:
   - Trains a linear regression model on encrypted data.
   - Decrypts the data for training purposes.

5. **evaluate_model(model, enc_X, enc_y)**:
   - Evaluates the linear regression model on encrypted data.

6. **generate_random_data(num_samples, num_features, noise=0.1)**:
   - Generates random data for training with specified number of samples, features, and noise.

7. **generate_valid_plain_moduli(poly_modulus_degree, count)**:
   - Generates a list of valid plaintext moduli for a given polynomial modulus degree.

8. **generate_poly_modulus_degrees(size)**:
   - Generates a list of polynomial modulus degrees.

#### Main Execution Block
- Defines the size of the polynomial modulus degrees and the count of plaintext moduli.
- Generates the parameter grid for polynomial modulus degrees and plaintext moduli.
- Performs a grid search to find the optimal parameters by:
  - Generating random data.
  - Encrypting the data.
  - Training and evaluating the linear regression model on encrypted data.
  - Tracking the parameters that yield the best MSE.

### Usage
To run the script, simply execute it with Python:
```bash
python homomorphic encryption optimization.py
```

### Output
The script will output the MSE for each parameter combination tested and will eventually print the best parameters and the corresponding MSE.

Example output:
```
Params: {'poly_modulus_degree': 4096, 'plain_modulus': 8193}, MSE: 0.1234
Params: {'poly_modulus_degree': 4096, 'plain_modulus': 12289}, MSE: 0.5678
...
Best Params: {'poly_modulus_degree': 8192, 'plain_modulus': 65537}, Best MSE: 0.0123
```

### Notes
- Ensure your machine has sufficient memory and computational power to handle the encryption and decryption processes, especially when using large polynomial modulus degrees.
- This script is a starting point and can be extended or modified to test other machine learning models, encryption schemes, or optimization techniques.
