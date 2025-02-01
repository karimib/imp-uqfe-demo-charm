from charm.toolbox.pairinggroup import PairingGroup, G1, G2
import time
import csv
from qfehelpers import random_vector, pp_size_bit, msk_size_bit, ct_size_bit, skf_size_bit
from uqfer import UQFE


group = PairingGroup("BN254")
p_order = group.order()
g1 = group.random(G1)
g1.initPP()
g2 = group.random(G2)
g2.initPP()
gt = group.pair_prod(g1, g2)
gt.initPP()


def implementation_check():
    k = 4
    k_prime = 4 
     # lenght of the key for the PRF in bytes
    lamda = 16

    
    pparam = True
    if pparam:
        print("PARAMETERS")
        print("k: ", k)
        print("k_prime: ", k_prime)
        print("lamda: ", lamda)
        
    G = UQFE(group, p_order, g1, g2, gt, k, k_prime, lamda)
    pp, msk = G.setup(p_order)

    psetup = False
    if psetup:
        print("A_0_G_1: ", pp.A_0_G_1)
        print("A_0_W_1_G_1: ", pp.A_0_W_1_G_1)
        print("A_0_W_2_G_1: ", pp.A_0_W_2_G_1)
        print("K_1: ", msk.K_1)
        print("K_2: ", msk.K_2)
        print("W_1: ", msk.W_1)
        print("W_2: ", msk.W_2)

    n1 = 100
    n2 = 100
    z_1 = random_vector(0, 2, n1)
    z_2 = random_vector(0, 2, n2)
    Iz_1 = [i for i in range(1, n1 + 1)]
    Iz_2 = [i for i in range(1, n2 + 1)]
    
    pencrypt = False
    if pencrypt:
        print("INPUTS")
        print("n1: ", n1)
        print("n2: ", n2)
        print("z_1: ", z_1)
        print("z_2: ", z_2)
        print("Iz_1: ", Iz_1)
        print("Iz_2: ", Iz_2)
    CT, CT_plain = G.encrypt(pp, msk, z_1, z_2, Iz_1, Iz_2)
    

    f = random_vector(1, 2, n1 * n2)
    If_1 = Iz_1
    If_2 = Iz_2
    pkeygen = False
    if pkeygen:
        print("FUNCTION")
        print("F: ", f)
        print("If_1: ", If_1)
        print("If_2: ", If_2)
    skf, skf_plain = G.keygen(pp, msk, f, If_1, If_2)

    v = G.decrypt(pp, skf_plain, CT_plain)
    print("expected result: ", G.get_expected_result(p_order, z_1, f, z_2))
    print("v: ", v)
    
    s_pp = pp_size_bit(group, pp) / 1024
    s_msk= msk_size_bit(group, msk) / 1024
    s_ct = ct_size_bit(group, CT) / 1024
    s_skf = skf_size_bit(group, skf) / 1024

    print("pp size in kbits: ", s_pp)
    print("msk size in kbits: ", s_msk)
    print("ct size in kbits: ", s_ct)
    print("skf size in kbits: ", s_skf)



# Simulation of the UQFE scheme with vectors of increasing length k and small values
def simulation_fixed_vectors():
    results = []
    # lenght of the key for the PRF in bytes
    lamda = 16
    z_1_max = 3
    z_2_max = 2
    f_max = 2

    for k in range(3, 65):
        k = k
        k_prime = k
        G = UQFE(group, p_order, g1, g2, gt, k, k_prime, lamda)
        
        

        start_time = time.time()
        pp, msk = G.setup(p_order)
        setup_time = time.time() - start_time
        
        n1 = 4
        n2 = 4
        z_1 = random_vector(0, z_1_max, n1)
        z_2 = random_vector(0, z_2_max, n2)
        Iz_1 = [i for i in range(1, n1 + 1)]
        Iz_2 = [i for i in range(1, n2 + 1)]
        f = random_vector(1, f_max, n1 * n2)
        If_1 = Iz_1
        If_2 = Iz_2

        start_time = time.time()
        CT, CT_plain = G.encrypt(pp, msk, z_1, z_2, Iz_1, Iz_2)
        encrypt_time = time.time() - start_time

        start_time = time.time()
        skf, skf_plain = G.keygen(pp, msk, f, If_1, If_2)
        keygen_time = time.time() - start_time

        start_time = time.time()
        v = G.decrypt(pp, skf_plain, CT_plain)
        decrypt_time = time.time() - start_time

        setup_time *= 1_000_000_000
        keygen_time *= 1_000_000_000
        encrypt_time *= 1_000_000_000
        decrypt_time *= 1_000_000_000
        total_time = setup_time + keygen_time + encrypt_time + decrypt_time

        #expected = G.get_expected_result(p_order, z_1, f, z_2)
        #print("expected result: ", expected)
        #print("calculated result: ", v)
        s_pp = pp_size_bit(group, pp) / 1024
        s_msk= msk_size_bit(group, msk) / 1024
        s_ct = ct_size_bit(group, CT) / 1024
        s_sk = skf_size_bit(group, skf) / 1024
        results.append(
            [
                k,
                k_prime,
                lamda,
                s_msk,
                s_pp,
                s_ct,
                s_sk,
                setup_time,
                keygen_time,
                encrypt_time,
                decrypt_time,
                total_time,
            ]
        )

    with open(
        "data/uqfe_benchmark_fixed_vectors_sizes_variable_k.csv", "w", newline=""
    ) as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "k",
                "k_prime",
                "lamda",
                "size msk",
                "size pp",
                "size ct",
                "size sk",
                "time setup",
                "time keygen",
                "time encrypt",
                "time decrypt",
                "time total",
            ]
        )
        csvwriter.writerows(results)


# Simulation of the UQFE scheme with vectors of fixed length k with values between 1 and p_order
def simulation_p_vectors():
    results = []
     
    # lenght of the key for the PRF in bytes
    lamda = 16
    z_1_max = 2
    z_2_max = 2
    f_max = 2

    
    k = 4
    k_prime = 4
    G = UQFE(group, p_order, g1, g2, gt, k, k_prime, lamda)

    start_time = time.time()
    pp, msk = G.setup(p_order)
    setup_time = time.time() - start_time
    setup_time *= 1_000_000_000
    
    for l in range(3,65):
        n1 = l
        n2 = n1
        z_1 = random_vector(0, z_1_max, n1)
        z_2 = random_vector(0, z_2_max, n2)
        Iz_1 = [i for i in range(1, n1 + 1)]
        Iz_2 = [i for i in range(1, n2 + 1)]
        f = random_vector(1, f_max, n1 * n2)
        If_1 = Iz_1
        If_2 = Iz_2

        start_time = time.time()
        CT, CT_Plain = G.encrypt(pp, msk, z_1, z_2, Iz_1, Iz_2)
        encrypt_time = time.time() - start_time

        start_time = time.time()
        skf, skf_plain = G.keygen(pp, msk, f, If_1, If_2)
        keygen_time = time.time() - start_time

        start_time = time.time()
        v = G.decrypt(pp, skf_plain, CT_Plain)
        decrypt_time = time.time() - start_time

        
        keygen_time *= 1_000_000_000
        encrypt_time *= 1_000_000_000
        decrypt_time *= 1_000_000_000
        total_time = setup_time + keygen_time + encrypt_time + decrypt_time

        expected = G.get_expected_result(p_order, z_1, f, z_2)
        print("expected result: ", expected)
        print("calculated result: ", v)
        s_pp = pp_size_bit(group, pp) / 1024
        s_msk= msk_size_bit(group, msk) / 1024
        s_ct = ct_size_bit(group, CT) / 1024
        s_sk = skf_size_bit(group, skf) / 1024

        results.append(
            [
                n1,
                n2,
                k,
                k_prime,
                lamda,
                s_msk,
                s_pp,
                s_ct,
                s_sk,
                setup_time,
                keygen_time,
                encrypt_time,
                decrypt_time,
                total_time,
            ]
        )

    with open("data/uqfe_benchmark_message_lengths_range.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [   "n1",
                "n2",
                "k",
                "k_prime",
                "lamda",
                "size msk",
                "size pp",
                "size ct",
                "size sk",
                "time setup",
                "time keygen",
                "time encrypt",
                "time decrypt",
                "time total",
            ]
        )
        csvwriter.writerows(results)


#simulation_fixed_vectors()
#simulation_p_vectors()
#implementation_check()
