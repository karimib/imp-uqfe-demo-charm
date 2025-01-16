
from charm.toolbox.pairinggroup import PairingGroup, GT, G1, G2
import time
import csv
from qfehelpers import (
    random_vector,
    random_int_matrix,
)
from uqfe import UQFE

k = 12
k_prime = 12
lamda = 128

# Benchmarking against
# k = 2 - 64
# lamda = 128, 256, 512, 1024





group = PairingGroup("BN254")
p_order = group.order()
g1 = group.random(G1)
g1.initPP()
g2 = group.random(G2)
g2.initPP()
gt = group.pair_prod(g1, g2)
G = UQFE(group, p_order, g1, g2, gt, k, k_prime, lamda)

def implementation_check():
    psetup = False
    pp, msk = G.setup(p_order)
    if psetup:
        print("A_0_G_1: ", pp.A_0_G_1)
        print("A_0_W_1_G_1: ", pp.A_0_W_1_G_1)
        print("A_0_W_2_G_1: ", pp.A_0_W_2_G_1)
        print("K_1: ", msk.K_1)
        print("K_2: ", msk.K_2)
        print("W_1: ", msk.W_1)
        print("W_2: ", msk.W_2)

    pencrypt = True
    z_1 = random_vector(1, 5, k)
    z_2 = random_vector(1, 5, k_prime)
    Iz_1 = [i for i in range(1, k+1)]
    Iz_2 = [i for i in range(1, k_prime+1)]
    print("INPUTS")
    print("z_1: ", z_1)
    print("z_2: ", z_2)
    print("Iz_1: ", Iz_1)
    print("Iz_2: ", Iz_2)
    CT, CT_Plain = G.encrypt(pp, msk, z_1, z_2, Iz_1, Iz_2)
    
    pkeygen = True
    f = random_vector(1, 2, k*k_prime)
    If_1 = Iz_1
    If_2 = Iz_2
    print("FUNCTION")
    print("F: ", f)
    print("If_1: ", If_1)
    print("If_2: ", If_2)
    skf, skf_plain = G.keygen(pp, msk, f, If_1, If_2)

    pdecrypt = True
    print("expected result: ", G.get_expected_result(p_order, z_1, f, z_2))
    v = G.decrypt(pp, skf_plain, CT_Plain)
    print("v: ", v)
    

       

# Inputs, Ciphertexts size, How expensive are the single steps
# Presentation
def simulation_fixed_vectors():
    results = []
    x_max = 3
    y_max = 2
    F_max = 2
    for k in range(3, 100):
        m = k
        n = k - 1
        x = random_vector(1, x_max, n)
        y = random_vector(1, y_max, m)
        F = random_int_matrix(1, F_max, n, m)
        p = p_order

        start_time = time.time()
        mpk, msk = G.setup(p, k)
        setup_time = time.time() - start_time

        start_time = time.time()
        skF = G.keygen(p, mpk, msk, F)
        keygen_time = time.time() - start_time


        start_time = time.time()
        CT_xy = G.encrypt(msk, x, y)
        encrypt_time = time.time() - start_time

        start_time = time.time()
        v = G.decrypt(p, mpk, skF, CT_xy, n, m, F)
        decrypt_time = time.time() - start_time

        setup_time *= 1_000_000_000
        keygen_time *= 1_000_000_000
        encrypt_time *= 1_000_000_000
        decrypt_time *= 1_000_000_000
        total_time = setup_time + keygen_time + encrypt_time + decrypt_time

        expected = G.get_expected_result(p, x, F, y)
        print("expected result: ", expected)
        print("calculated result: ", v)

        results.append([k, m, n, x_max, y_max, F_max, setup_time, keygen_time, encrypt_time, decrypt_time, total_time])

    with open('data/qfe_benchmark_fixed_vectors.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['k', 'm', 'n', 'x_max', 'y_max', 'F_max' ,'time setup', 'time keygen', 'time encrypt', 'time decrypt', 'time total'])
        csvwriter.writerows(results)



def simulation_p_vectors():
    results = []
    x_max = p_order
    y_max = p_order
    F_max = p_order
    for k in range(3, 4):
        m = k
        n = k - 1
        x = random_vector(1, x_max, n)
        y = random_vector(1, y_max, m)
        F = random_int_matrix(1, F_max, n, m)
        p = p_order

        start_time = time.time()
        mpk, msk = G.setup(p, k)
        setup_time = time.time() - start_time

        start_time = time.time()
        skF = G.keygen(p, mpk, msk, F)
        keygen_time = time.time() - start_time


        start_time = time.time()
        CT_xy = G.encrypt(msk, x, y)
        encrypt_time = time.time() - start_time

        start_time = time.time()
        v = G.decrypt(p, mpk, skF, CT_xy, n, m, F)
        decrypt_time = time.time() - start_time

        setup_time *= 1_000_000_000
        keygen_time *= 1_000_000_000
        encrypt_time *= 1_000_000_000
        decrypt_time *= 1_000_000_000
        total_time = setup_time + keygen_time + encrypt_time + decrypt_time

        expected = G.get_expected_result(p, x, F, y)
        print("expected result: ", expected)
        print("calculated result: ", v)

        results.append([k, m, n, x_max, y_max, F_max, setup_time, keygen_time, encrypt_time, decrypt_time, total_time])

    with open('data/qfe_benchmark_p_vectors.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['k', 'm', 'n', 'x_max', 'y_max', 'F_max' ,'time setup', 'time keygen', 'time encrypt', 'time decrypt', 'time total'])
        csvwriter.writerows(results)



#simulation_fixed_vectors()
#simulation_p_vectors()
implementation_check()
