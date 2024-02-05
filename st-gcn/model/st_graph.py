import numpy as np

class ST_Graph:
    pass

def symnorm(A):
    """Compute a symmetrically normalized 
    adjacency matrix

    Args:
        A : adjacency matrix with self-loop

    Returns:
        A_symnorm: symnormed A
    """
    D = np.diag(np.sum(A, axis=0))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    return D_inv_sqrt @ A @ D_inv_sqrt

def get_adjacency(edges, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A

if __name__ == '__main__':
    # Test input
    num_nodes = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 1)]

    A = get_adjacency(edges, num_nodes)
    print(A)
    print(symnorm(A))
    print(np.sum(symnorm(A), axis=1))