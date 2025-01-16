import secrets
import hmac
import hashlib
from charm.toolbox.pairinggroup import PairingGroup, G1, G2, GT
from charm.core.math.pairing import pair
from qfehelpers import (
    apply_to_matrix,
    tensor_product,
    random_int_matrix,
    matrix_multiply_mod,
    generate_random_key,
    pseudo_random_function,
    matrix_concat,
    random_vector,
    vector_matrix_multiply_mod,
    get_matrix_dimensions,
    apply_to_vector,
    add_vectors,
    identity_matrix,
    transpose_matrix,
    transpose_vector,
    tensor_product_matrix_vector,
    tensor_product_vectors,
    dot_product,
    PP,
    MSK,
    SKF,
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
    HF_1 = None
    HF_2 = None
    AF_1 = None
    AF_2 = None

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
        self.HF_1 = []
        self.HF_2 = []
        self.AF_1 = []
        self.AF_2 = []
        

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
            #a1 = [2,3,4]
            a1 = random_vector(1, self.p_order, self.k, seed=i_l)
            self.H_1.append((apply_to_vector(a1, self.g1), apply_to_vector(a1, self.g2)))
            self.A_1.append(a1)

        for j_l in Iz_2:
            #a2 = [4,5,7]
            a2 = random_vector(1, self.p_order, self.k_prime, seed=j_l)
            self.H_2.append(apply_to_vector(a2, self.g2))
            self.A_2.append(a2)

        w_1 = []
        w_2 = []
        for i_l in Iz_1:
            w_1.append(pseudo_random_function(msk.K_1, i_l))

        for j_l in Iz_2:
            w_2.append(pseudo_random_function(msk.K_2, j_l))

        w_1 = [w_1]
        w_2 = [w_2]
        
        W_1_tilde = tensor_product(msk.W_1, w_1)
        W_2_tilde = tensor_product(msk.W_2, w_2)


        A_0_W = matrix_concat(matrix_multiply_mod(self.A_0, W_1_tilde, self.p_order), matrix_multiply_mod(self.A_0, W_2_tilde, self.p_order))
        s_1 = random_vector(0, self.p_order, self.k) 
        s_0 = random_vector(0, self.p_order, self.k_prime) 
        s_2 = random_vector(0, self.p_order, self.k_prime) 
     
        y1 = add_vectors(vector_matrix_multiply_mod(s_1, self.A_1, self.p_order), z_1)
        y2 = add_vectors(vector_matrix_multiply_mod(s_2, self.A_2, self.p_order), z_2)
        c0 = vector_matrix_multiply_mod(s_0, self.A_0, self.p_order)
        y0 = vector_matrix_multiply_mod(s_0, A_0_W, self.p_order)
        t0 = tensor_product([s_1], [z_2])
        #print("y0: ", y0)
        #print("t0: ", t0)
        t1 = tensor_product([y1], [s_2])
        #print("t1: ", t1)
        t2 = matrix_concat(t0,t1)
        #print("t2: ", t2)
        t2 = [element for sublist in t2 for element in sublist]
        y0 = add_vectors(y0, t2)
        #print(y0)

        ct_plain = CT(y1, y2, c0, y0, Iz_1, Iz_2)
        y1 = apply_to_vector(y1, self.g1)
        y2 = apply_to_vector(y2, self.g2)
        c0 = apply_to_vector(c0, self.g1)
        y0 = apply_to_vector(y0, self.g1)

        ct = CT(y1, y2, c0, y0, Iz_1, Iz_2)
        return ct, ct_plain

    def keygen(self, pp, msk, f, If_1, If_2):
        for i_l in If_1:
            af1 = random_vector(1, self.p_order, self.k, seed=i_l)
            #af1 = [2,3,4]
            self.HF_1.append((apply_to_vector(af1, self.g1), apply_to_vector(af1, self.g2)))
            self.AF_1.append(af1)


        for j_l in If_2:
            # Fixing the seed for testing purposes
            af2 = random_vector(1, self.p_order, self.k_prime,seed=j_l)
            #af2 = [4,5,7]
            self.HF_2.append(apply_to_vector(af2, self.g2))
            self.AF_2.append(af2)
        
        wf_1 = []
        wf_2 = []
        for i_l in If_1:
            wf_1.append(pseudo_random_function(msk.K_1, i_l))

        for j_l in If_2:
            wf_2.append(pseudo_random_function(msk.K_2, j_l))

      
        t0 = tensor_product_matrix_vector(msk.W_1, wf_1)
        #get_matrix_dimensions(t0, "t0: ")
        t1 = tensor_product_matrix_vector(msk.W_2, wf_2)
        #get_matrix_dimensions(t1, "t1: ")
        WF = matrix_concat(t0,t1)
        #get_matrix_dimensions(WF, "WF: ")

        AFF = tensor_product(self.AF_1, identity_matrix(len(If_2)))
        AFF = matrix_multiply_mod(AFF, transpose_vector(f), self.p_order)
        #get_matrix_dimensions(AFF, "AFF: ")
        BFF = tensor_product(identity_matrix(len(If_1)), self.AF_2)
        BFF = matrix_multiply_mod(BFF, transpose_vector(f), self.p_order)
        #get_matrix_dimensions(BFF, "BFF: ")
        CFF = AFF + BFF
        #get_matrix_dimensions(CFF, "CFF: ")
        k1 = matrix_multiply_mod(WF, CFF, self.p_order)
        #get_matrix_dimensions(k1, "k1: ")
        sk_plain = SKF(k1, f, If_1, If_2)
        k1 = apply_to_matrix(k1, self.g2)
        sk = SKF(k1, f, If_1, If_2)
        
        return sk, sk_plain

    def decrypt(self, pp, skf, ct):
        if (ct.Iz_1 != skf.If_1) or (ct.Iz_2 != skf.If_2):
            print("Error: The ciphertext and the secret key are not compatible")
            return None
        
        r0 = tensor_product(self.A_1, identity_matrix(len(ct.Iz_2)))
        r0 = matrix_multiply_mod(r0, transpose_vector(skf.F), self.p_order)
        #get_matrix_dimensions(r0, "r0: ")
        r1 = tensor_product(identity_matrix(len(ct.Iz_1)), self.A_2)
        r1 = matrix_multiply_mod(r1, transpose_vector(skf.F), self.p_order)
        #get_matrix_dimensions(r1, "r1: ")

        k2 = r0 + r1
        k2 = [element for sublist in k2 for element in sublist]
        #print("r2: ", k2)
        
        k1 = [element for sublist in skf.k1 for element in sublist]
        e1 = dot_product(ct.c_0, k1) % self.p_order
        #print("e1: ", e1)

        e2 = dot_product(ct.y_0, k2) % self.p_order
        #print("e2: ", e2)

        e0 = tensor_product_vectors(ct.y_1, ct.y_2)
        e0 = dot_product(e0, skf.F) % self.p_order
        #print("e0: ", e0)
        d = self.gt ** e0
        d *= self.gt ** e1
        d *= -(self.gt ** e2)
        #print("d: ", d)

        v = 0
        res = self.group.random(GT)
        while d != res and v < self.p_order:
            v += 1
            res = self.gt ** int(v)

        return v


    def get_expected_result(self, p_order, z_1, f, z_2):
        res = tensor_product_vectors(z_1, z_2)
        res = dot_product(res, f) % p_order
        return res