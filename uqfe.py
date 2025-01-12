import secrets
import hmac
import hashlib
from charm.toolbox.pairinggroup import PairingGroup, G1, G2, GT
from qfehelpers import (
    apply_to_matrix,
    tensor_product,
    random_int_matrix,
    matrix_multiply_mod,
    generate_random_key,
    pseudo_random_function,
    matrix_concat,
    get_matrix_dimensions,
    PP,
    MSK,
    CT
)

######################################## PARAMETERS ###################################################################

# Initialize a set of parameters from a string
# check the PBC documentation (http://crypto.stanford.edu/pbc/manual/) for more information
#
## Type f pairing
# TODO: Q: How to find parameters for thes pairing ? Are these parameters a dependency of choosing k ? Or F, n, m ?
# TODO: BM: Benchmark maybe over different curves ?
# TODO: Find curve of Type III -> https://arxiv.org/pdf/1908.05366
# TODO: Find such a curve: https://www.cryptojedi.org/papers/pfcpo.pdf
# TODO: Different sizes of ciphertexts
# TODO: 5 different types of experiments


# Initialize the pairing
# group = PairingGroup("BN254")

# TODO: Find p > mnB_xB_yB_f where M: {0,...,B_x}^n x {0,...B_y}^m and K:={0,..,B_f}^nxm to efficiently compute dlog P8,9 QFE Paper
# TODO: BM: Maybe do benchmarks over different sizes of p
# TODO: Outsource parameters in files
# p = group.order() # prime for group Z_p
# p_order = group.order()
# k = 9  # parameter for generation of D-k matrices
# m = k
# n = k - 1
# x = random_vector(1, 3, n)
# y = random_vector(1, 2, m)
# F = random_int_matrix(1, 2, n, m)

######################################## ALGORITHM ####################################################################


# 1. Add to Mult
# 2. Search curves
# 3. Search other implementations
# TODO: Remove after bechmarking
class UQFE:
    # TODO: Remove after bechmarking
    group = None
    p_order = None
    g1 = None
    g2 = None
    gt = None
    k = None
    k_prime = None
    lamda = None
    H_1 = None
    H_2 = None
    A_0 = None
    A_1 = None
    A_2 = None

    # TODO: Remove after bechmarking
    def __init__(self, group, p_order, g1, g2, gt, k, k_prime, lamda):
        self.group = group
        self.p_order = p_order
        self.g1 = g1
        self.g2 = g2
        self.gt = gt
        self.k = k
        self.k_prime = k_prime
        self.lamda = lamda
        self.H_1 = []
        self.H_2 = []
        self.A_1 = []
        self.A_2 = []

    def get_p_order(self):
        return self.p_order

    def setup(self, p=p_order):
        # Parameters

        A_0 = random_int_matrix(0, p, self.k_prime, self.k_prime + 1)
        self.A_0 = A_0
        W_1 = random_int_matrix(0, p, self.k_prime + 1, self.k_prime)
        W_2 = random_int_matrix(0, p, self.k_prime + 1, self.k)
        # Q: Are these keys the seeds for the PRF ?
        K_1 = generate_random_key(self.lamda)
        K_2 = generate_random_key(self.lamda)
        A_0_G_1 = apply_to_matrix(A_0, self.g1)
        A_0_W_1_G_1 = apply_to_matrix(matrix_multiply_mod(A_0, W_1, p), self.g1)
        A_0_W_2_G_1 = apply_to_matrix(matrix_multiply_mod(A_0, W_2, p), self.g1)
        pp = PP(self.group, A_0_G_1, A_0_W_1_G_1, A_0_W_2_G_1)
        msk = MSK(K_1, K_2, W_1, W_2)
        return pp, msk

    def encrypt(self, pp, msk, z_1, z_2, Iz_1, Iz_2):
        for i_l in Iz_1:
            a1 = random_int_matrix(1, self.p_order, self.k, 1)
            self.H_1.append((apply_to_matrix(a1, self.g1), apply_to_matrix(a1, self.g2)))
            self.A_1.append(a1)

        for j_l in Iz_2:
            a2 = random_int_matrix(1, self.p_order, self.k_prime, 1)
            self.H_2.append(apply_to_matrix(a2, self.g2))
            self.A_2.append(a2)

        w_1 = []
        w_2 = []
        for i_l in Iz_1:
            w_1.append([pseudo_random_function(msk.K_1, i_l)])

        for j_l in Iz_2:
            w_2.append([pseudo_random_function(msk.K_2, j_l)])

        print("k : ", self.k)
        print("k_prime : ", self.k_prime)
        print("n1 : ", len(self.H_1))
        print("n2 : ", len(self.H_2))
        print("w1 : ", w_1)
        W_2_tilde = tensor_product(msk.W_2, w_2)
        W_1_tilde = tensor_product(msk.W_1, w_1)
        # TODO: Check if this is correct
        A_O_W = matrix_concat(matrix_multiply_mod(self.A_0, W_1_tilde, self.p_order), matrix_multiply_mod(self.A_0, W_2_tilde, self.p_order))
        s_1 = random_int_matrix(1, self.p_order, self.k, 1)
        s_0 = random_int_matrix(1, self.p_order, self.k_prime, 1)
        s_2 = random_int_matrix(1, self.p_order, self.k_prime, 1)

        y1 = matrix_multiply_mod(s_1, self.A_1, self.p_order) + z_1
        y2 = matrix_multiply_mod(s_2, self.A_2, self.p_order) + z_2
        c0 = matrix_multiply_mod(s_0, self.A_0, self.p_order)
        y0 = matrix_multiply_mod(s_0, A_O_W, self.p_order) + tensor_product(s_1, z_2) + tensor_product(y1, s_2)

        ct = CT(y1, y2, c0, y0)


        return ct

    def keygen(self, p=p_order, mpk=None, msk=None, F=None):
        
        return 0

    def decrypt(
        self, p=p_order, mpk=None, skF=None, CT_xy=None, n=None, m=None, F=None
    ):
        

        return v
