from charm.toolbox.pairinggroup import GT
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
    apply_to_vector,
    add_vectors,
    identity_matrix,
    transpose_vector,
    tensor_product_matrix_vector,
    tensor_product_vectors,
    dot_product,
    PP,
    MSK,
    SKF,
    CT,
)

########################################
# Helper to transpose a list-of-lists
# so that rows become columns and vice versa.
########################################
def transpose_2d(matrix_2d):
    """
    matrix_2d is a list of lists, each "row" the same length.
    Returns the transposed list of lists.
    """
    return list(map(list, zip(*matrix_2d)))

######################################## ALGORITHM ####################################################################

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

    # These will hold the per-index hashing vectors (and are turned into matrices below)
    H_1 = None
    H_2 = None
    A_0 = None
    # Instead of storing A_1, A_2 as final matrices, we store them as lists of row vectors
    # so that A_1_rows[i] is the row for index i, each row length k or k'
    A_1_rows = None  ### CHANGED
    A_2_rows = None  ### CHANGED

    HF_1 = None
    HF_2 = None
    AF_1_rows = None  ### CHANGED
    AF_2_rows = None  ### CHANGED

    def __init__(self, group, p_order, g1, g2, gt, k, k_prime, lamda):
        self.group = group
        self.p_order = p_order
        self.g1 = g1
        self.g2 = g2
        self.gt = gt
        self.k = k
        self.k_prime = k_prime
        self.lamda = lamda

        # We will build up these lists on-the-fly
        self.H_1 = []
        self.H_2 = []
        self.A_1_rows = []  ### CHANGED
        self.A_2_rows = []  ### CHANGED

        self.HF_1 = []
        self.HF_2 = []
        self.AF_1_rows = []  ### CHANGED
        self.AF_2_rows = []  ### CHANGED

    def get_p_order(self):
        return self.p_order

    def setup(self, p=None):
        if p is None:
            p = self.p_order
        # A_0 is kPrime x (kPrime+1)
        A_0 = random_int_matrix(0, p, self.k_prime, self.k_prime + 1)
        self.A_0 = A_0

        # W_1 is (kPrime+1) x kPrime
        W_1 = random_int_matrix(0, p, self.k_prime + 1, self.k_prime)
        # W_2 is (kPrime+1) x k
        W_2 = random_int_matrix(0, p, self.k_prime + 1, self.k)

        K_1 = generate_random_key(self.lamda)
        K_2 = generate_random_key(self.lamda)

        # Convert A_0 etc. into G1 / G2 form for the public parameters
        A_0_G_1 = apply_to_matrix(A_0, self.g1)
        A_0_W_1_G_1 = apply_to_matrix(matrix_multiply_mod(A_0, W_1, p), self.g1)
        A_0_W_2_G_1 = apply_to_matrix(matrix_multiply_mod(A_0, W_2, p), self.g1)

        pp = PP(self.group, A_0_G_1, A_0_W_1_G_1, A_0_W_2_G_1)
        msk = MSK(K_1, K_2, W_1, W_2)
        return pp, msk

    def encrypt(self, pp, msk, z_1, z_2, Iz_1, Iz_2):
        # Clear out old row data if re-using object multiple times
        self.H_1.clear()
        self.H_2.clear()
        self.A_1_rows.clear()  ### CHANGED
        self.A_2_rows.clear()  ### CHANGED

        # Build up A_1_rows (n1 x k) and H_1 (the group-lift)
        for i_l in Iz_1:
            # Each row is dimension k
            a1 = random_vector(1, self.p_order, self.k, seed=i_l)
            self.A_1_rows.append(a1)  ### CHANGED
            # Also store the lifted version in G1, G2
            self.H_1.append((apply_to_vector(a1, self.g1),
                             apply_to_vector(a1, self.g2)))

        # Build up A_2_rows (n2 x k') and H_2
        for j_l in Iz_2:
            a2 = random_vector(1, self.p_order, self.k_prime, seed=j_l)
            self.A_2_rows.append(a2)  ### CHANGED
            self.H_2.append(apply_to_vector(a2, self.g2))

        # Convert list-of-rows into actual matrices so we can do matrix multiply
        # A_1_mat is shape (k x n1) because each row is length k -> transpose
        A_1_mat = transpose_2d(self.A_1_rows)  ### CHANGED
        # A_2_mat is shape (k' x n2)
        A_2_mat = transpose_2d(self.A_2_rows)  ### CHANGED

        # Build PRF outputs
        w_1 = []
        for i_l in Iz_1:
            w_1.append(pseudo_random_function(msk.K_1, i_l))
        w_2 = []
        for j_l in Iz_2:
            w_2.append(pseudo_random_function(msk.K_2, j_l))

        # W_1_tilde and W_2_tilde: tensor with the index-based w_1, w_2
        W_1_tilde = tensor_product_matrix_vector(msk.W_1, w_1)
        W_2_tilde = tensor_product_matrix_vector(msk.W_2, w_2)

        # Concatenate the two big vectors
        A_0_W = matrix_concat(
            matrix_multiply_mod(self.A_0, W_1_tilde, self.p_order),
            matrix_multiply_mod(self.A_0, W_2_tilde, self.p_order),
        )

        # Ephemeral randomness: dimension k or k'
        s_1 = random_vector(0, self.p_order, self.k)      # shape k
        s_0 = random_vector(0, self.p_order, self.k_prime)# shape k'
        s_2 = random_vector(0, self.p_order, self.k_prime)# shape k'

        # y1 = s_1 * A_1_mat + z_1
        # A_1_mat is (k x n1), so result is dimension n1
        partial_y1 = vector_matrix_multiply_mod(s_1, A_1_mat, self.p_order)
        y1 = add_vectors(partial_y1, z_1)  ### dimension n1

        # y2 = s_2 * A_2_mat + z_2
        # A_2_mat is (k' x n2), so result is dimension n2
        partial_y2 = vector_matrix_multiply_mod(s_2, A_2_mat, self.p_order)
        y2 = add_vectors(partial_y2, z_2)  ### dimension n2

        # c0 = s_0 * A_0
        c0 = vector_matrix_multiply_mod(s_0, self.A_0, self.p_order)

        # y0 = s_0 * (A_0_W) initially
        y0 = vector_matrix_multiply_mod(s_0, A_0_W, self.p_order)

        # Build the additional “tensor” terms
        # t0 = (s_1 x z_2)
        # t1 = (y1  x s_2)
        # Then flatten and add
        t0 = tensor_product([s_1], [z_2])   # shape k x n2, flattened
        t1 = tensor_product([y1], [s_2])   # shape n1 x k', flattened
        t2 = matrix_concat(t0, t1)
        t2 = [element for sublist in t2 for element in sublist]  # flatten
        y0 = add_vectors(y0, t2)

        # Return a plaintext version and a group-lifted version
        ct_plain = CT(y1, y2, c0, y0, Iz_1, Iz_2)

        # Now apply group-lifting
        y1_g1 = apply_to_vector(y1, self.g1)
        y2_g2 = apply_to_vector(y2, self.g2)
        c0_g1 = apply_to_vector(c0, self.g1)
        y0_g1 = apply_to_vector(y0, self.g1)

        ct = CT(y1_g1, y2_g2, c0_g1, y0_g1, Iz_1, Iz_2)
        return ct, ct_plain

    def keygen(self, pp, msk, f, If_1, If_2):
        # Clear old data in case re-using object
        self.HF_1.clear()
        self.HF_2.clear()
        self.AF_1_rows.clear()  ### CHANGED
        self.AF_2_rows.clear()  ### CHANGED

        # Build the row-lists
        for i_l in If_1:
            af1 = random_vector(1, self.p_order, self.k, seed=i_l)
            self.AF_1_rows.append(af1)  ### CHANGED
            self.HF_1.append((apply_to_vector(af1, self.g1),
                              apply_to_vector(af1, self.g2)))

        for j_l in If_2:
            af2 = random_vector(1, self.p_order, self.k_prime, seed=j_l)
            self.AF_2_rows.append(af2)  ### CHANGED
            self.HF_2.append(apply_to_vector(af2, self.g2))

        # Convert to (k x len(If_1)) and (k' x len(If_2)) form
        AF_1_mat = transpose_2d(self.AF_1_rows)  ### shape (k x |If_1|)
        AF_2_mat = transpose_2d(self.AF_2_rows)  ### shape (k' x |If_2|)

        # Build PRF outputs
        wf_1 = []
        for i_l in If_1:
            wf_1.append(pseudo_random_function(msk.K_1, i_l))
        wf_2 = []
        for j_l in If_2:
            wf_2.append(pseudo_random_function(msk.K_2, j_l))

        # Combine W_1, wf_1, etc.
        t0 = tensor_product_matrix_vector(msk.W_1, wf_1)
        t1 = tensor_product_matrix_vector(msk.W_2, wf_2)
        WF = matrix_concat(t0, t1)

        # AFF = (AF_1_mat) \otimes I_{|If_2|}, then times f
        # BFF = I_{|If_1|} \otimes (AF_2_mat), then times f
        # But first, we do the straightforward approach:
        AFF = tensor_product(self.AF_1_rows, identity_matrix(len(If_2)))
        # Convert AFF to actual 2D and multiply by f
        AFF_2d = transpose_2d(AFF)  # shape (k * |If_1|) x (something), depending on the helper
        AFF_2d = matrix_multiply_mod(AFF_2d, transpose_vector(f), self.p_order)

        BFF = tensor_product(identity_matrix(len(If_1)), self.AF_2_rows)
        BFF_2d = transpose_2d(BFF)
        BFF_2d = matrix_multiply_mod(BFF_2d, transpose_vector(f), self.p_order)

        # Sum them up
        # If needed, ensure they’re the same shape
        CFF = AFF_2d + BFF_2d
        # Multiply by WF
        k1 = matrix_multiply_mod(WF, CFF, self.p_order)

        sk_plain = SKF(k1, f, If_1, If_2)
        k1_g2 = apply_to_matrix(k1, self.g2)
        sk = SKF(k1_g2, f, If_1, If_2)
        return sk, sk_plain

    def decrypt(self, pp, skf, ct):
        # Check that index sets match
        if (ct.Iz_1 != skf.If_1) or (ct.Iz_2 != skf.If_2):
            print("Error: The ciphertext and the secret key are not compatible")
            return None

        # Build the A_1, A_2 matrices exactly as in encryption
        # (We assume they were built and stored in self.A_1_rows, self.A_2_rows)
        # A_1 is shape (k x len(Iz_1))
        A_1_mat = transpose_2d(self.A_1_rows)
        # A_2 is shape (k' x len(Iz_2))
        A_2_mat = transpose_2d(self.A_2_rows)

        # r0 = (A_1 x I_{|Iz_2|}) * F
        #    = tensor_product(A_1, I_{n2}) * f
        # But we have A_1_rows in a python list-of-lists; do as the code does
        r0_block = tensor_product(self.A_1_rows, identity_matrix(len(ct.Iz_2)))
        r0_mat   = transpose_2d(r0_block)
        r0_res   = matrix_multiply_mod(r0_mat, transpose_vector(skf.F), self.p_order)

        # r1 = (I_{|Iz_1|} x A_2) * F
        r1_block = tensor_product(identity_matrix(len(ct.Iz_1)), self.A_2_rows)
        r1_mat   = transpose_2d(r1_block)
        r1_res   = matrix_multiply_mod(r1_mat, transpose_vector(skf.F), self.p_order)

        k2_list = r0_res + r1_res
        # Flatten
        k2 = [elem for row in k2_list for elem in row]

        # Flatten skf.k1 as well
        k1 = [element for sublist in skf.k1 for element in sublist]

        e1 = dot_product(ct.c_0, k1) % self.p_order
        e2 = dot_product(ct.y_0, k2) % self.p_order

        # e0 = <(y1 x y2), F>
        e0_vec = tensor_product_vectors(ct.y_1, ct.y_2)
        e0 = dot_product(e0_vec, skf.F) % self.p_order

        d = self.gt ** e0
        d *= self.gt ** e1
        d *= -(self.gt ** e2)

        # Solve discrete log by naive search (only feasible for small p)
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