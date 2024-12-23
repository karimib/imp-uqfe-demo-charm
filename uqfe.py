import random
from charm.toolbox.pairinggroup import PairingGroup, G1, G2, GT
from qfehelpers import (
    apply_to_matrix,
    transpose_matrix,
    random_int_matrix,
    matrix_multiply_mod,
    vector_matrix_multiply_mod,
    vector_multiply_mod,
    matrix_vector_multiply,
    vector_transposed_mul_matrix_mul_vector,
    dot_product,
    scalar_multiply,
    generate_random_key,
    PP,
    MPK,
    MSK,
    SKF,
    CTXY,
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

    # TODO: Remove after bechmarking
    def __init__(self, group, p_order, g1, g2, gt):
        self.group = group
        self.p_order = p_order
        self.g1 = g1
        self.g2 = g2
        self.gt = gt

    def get_p_order(self):
        return self.p_order

    def setup(self, p=p_order, k=None, lamda=None):
        A0 = random_int_matrix(0,p,k,k+1)
        W1 = random_int_matrix(0,p,k+1,k)
        W2 = random_int_matrix(0,p,k,k+1)
        K1 = generate_random_key(lamda)
        K2 = generate_random_key(lamda)
        A01 = apply_to_matrix(A0, self.g1)
        A0W1 = apply_to_matrix(matrix_multiply_mod(A0, W1, p), self.g1)
        A0W2 = apply_to_matrix(matrix_multiply_mod(A0, W2, p), self.g1)
        pp = PP(self.group, A01, A0W1, A0W2)
        msk = MSK(K1, K2, W1, W2)
        return pp, msk

    def keygen(self, p=p_order, mpk=None, msk=None, F=None):
        # Generate random element u <- Z_p
        u = random.randint(0, p - 1)  # u <- Z_p
        # Generate random matrix F <- Z_p^(n x m)
        A = msk.A
        B = msk.B
        r = msk.r
        s = msk.s
        g1 = mpk.g1
        g2 = mpk.g2
        n = len(r)
        m = len(s)

        sum = 0
        ATB = matrix_multiply_mod(transpose_matrix(A), B, p)
        for i in range(n):
            riT_AT_B = vector_matrix_multiply_mod(r[i], ATB, p)
            for j in range(m):
                riT_AT_B_sj = vector_multiply_mod(riT_AT_B, s[j], p)
                sum += (F[i][j] * riT_AT_B_sj) % p

        # Compute K and K_tilde
        K = g1 ** int(sum - u)
        K_tilde = g2 ** int(u)

        skF = SKF(K, K_tilde)  # secret key for F
        return skF

    def encrypt(self, msk, x, y):
        A = msk.A
        B = msk.B
        a = msk.a
        b = msk.b
        r = msk.r
        s = msk.s

        # Compute c and c_tilde
        c = [
            matrix_vector_multiply(A, r[i]) + scalar_multiply(b, x[i])
            for i in range(len(x))
        ]
        c_tilde = [
            matrix_vector_multiply(B, s[j]) + scalar_multiply(a, y[j])
            for j in range(len(y))
        ]

        CT_xy = CTXY(c, c_tilde)
        return CT_xy

    def decrypt(
        self, p=p_order, mpk=None, skF=None, CT_xy=None, n=None, m=None, F=None
    ):
        c = CT_xy.c
        c_tilde = CT_xy.c_tilde
        K = skF.K
        K_tilde = skF.K_tilde
        g1 = mpk.g1
        g2 = mpk.g2 
        gt = mpk.gt

        D = self.group.random(GT)
        exp = 0
        for i in range(n):
            for j in range(m):
                exp += int(F[i][j] * int(dot_product(c[i], c_tilde[j])))

        D = gt**exp
        D *= -(self.group.pair_prod(K, g2))
        D *= -(self.group.pair_prod(g1, K_tilde))

        # Find v such that [v * (b.T)*a]_T = D
        v = 0
        res = self.group.random(GT)
        inner = mpk.baT
        while D != res and v < p:
            v += 1
            res = gt ** int(v * inner)

        return v

    def get_expected_result(self,p=p_order, x=None, F=None, y=None):
        expected = vector_transposed_mul_matrix_mul_vector(x, F, y, p)
        return expected
