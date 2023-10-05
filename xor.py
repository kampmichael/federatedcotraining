import numpy as np

def xor_mechanism(matrix, epsilon, reference_matrix = None, copies = 1):
    ##
    # --- Parameters ---
    #  matrix:             Input BINARY matrix
    #  epsilon:            Privacy budget
    #  reference_matrix:   Reference BINARY matrix for noise generation.
    #                      Columns of matrix and reference_matrix should match, 
    #                      i.e., matrix.shape[1] = reference_matrix.shape[1]
    #  copies:             # of copies to generate (for saving time)
    #
    # --- Return ---
    #  released BINARY matrix/matrices
    
    eps = np.finfo(float).eps
    
    matrix = np.array(matrix)
    nrows, ncols = matrix.shape
    sens = 3000
    
    released = []
    

    if reference_matrix is not None:
        A = np.array(reference_matrix)

        A_nrows, A_ncols = A.shape


        M_11 = A.T @ A
        ones = np.ones((A_nrows, A_ncols))
        M_01 = ones.T @ A - M_11
        M_10 = A.T @ ones - M_11
        M_00 = nrows * np.ones((A_ncols, A_ncols)) - M_01 - M_11 - M_10
        M_1 = A.sum(axis=0)
        M_0 = A_nrows - M_1
        
        theta_tilde = np.log((M_11 * M_00 + eps) / (M_10 * M_01 + eps))
        diag_val = np.log((M_1 + eps) / (M_0 + eps))
        theta_tilde = theta_tilde - np.diag(np.diag(theta_tilde)) + np.diag(diag_val)
    else:
        theta_tilde = np.eye(ncols)
        
    f_theta = np.linalg.norm(theta_tilde, 'fro')
    theta = (epsilon / (sens * f_theta)) * theta_tilde
    
    off_diagonal = theta - np.diag(np.diag(theta))
    min_value = 2 * np.sum((off_diagonal < 0) * off_diagonal, axis=1) + np.diag(theta)
    max_value = 2 * np.sum((off_diagonal > 0) * off_diagonal, axis=1) + np.diag(theta)
    coeff = np.exp(min_value) - 1
    bound = (coeff < 0) * min_value + (coeff > 0) * max_value
    B_probs = (1 + coeff / (1 + np.exp(bound))) / 2

    for idx in range(copies):
        B = np.zeros((nrows, ncols), dtype=int)
        for j in range(nrows):
            B[j] = np.random.binomial(n=1, p=B_probs, size = (ncols))
        xor_matrix = np.logical_xor(matrix,B).astype(int)
        released.append(xor_matrix)
        print("Finished generating #%d." % idx)
    return released[0] if copies == 1 else released
            

def xor_noise(matrix, epsilon, copies = 1):  
    eps = np.finfo(float).eps
    matrix = np.array(matrix)
    nrows, ncols = matrix.shape
    sens = 3000
    released = []
    theta_tilde = np.eye(ncols)  
    f_theta = np.linalg.norm(theta_tilde, 'fro')
    theta = (epsilon / (sens * f_theta)) * theta_tilde
    off_diagonal = theta - np.diag(np.diag(theta))
    min_value = 2 * np.sum((off_diagonal < 0) * off_diagonal, axis=1) + np.diag(theta)
    max_value = 2 * np.sum((off_diagonal > 0) * off_diagonal, axis=1) + np.diag(theta)
    coeff = np.exp(min_value) - 1
    bound = (coeff < 0) * min_value + (coeff > 0) * max_value
    B_probs = (1 + coeff / (1 + np.exp(bound))) / 2
    for idx in range(copies):
        B = np.zeros((nrows, ncols), dtype=int)
        for j in range(nrows):
            B[j] = np.random.binomial(n=1, p=B_probs, size = (ncols))
        xor_matrix = np.logical_xor(matrix,B).astype(int)
        released.append(xor_matrix)
        #print("Finished generating #%d." % idx)
    return released[0] if copies == 1 else released