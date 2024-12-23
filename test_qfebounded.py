import time, unittest
from uqfe import setup, keygen, encrypt, decrypt

class QFEPureDemoBenchmarkTest(unittest.TestCase):
        def benchmark_method(self, method, *args):
            start_time = time.time()
            result = method(*args)
            end_time = time.time()
            return end_time - start_time, result

        def test_setup(self):
            sizes = [128, 256, 512]
            for k in sizes:
                duration, _ = self.benchmark_method(setup, k)
                print(f"Setup with size {k} took {duration:.6f} seconds")
                self.assertGreaterEqual(duration, 0)

        def test_keygen(self):
            sizes = [128, 256, 512]
            for k in sizes:
                _, params = self.benchmark_method(setup, k)
                duration, _ = self.benchmark_method(keygen, params)
                print(f"Keygen with size {k} took {duration:.6f} seconds")
                self.assertGreaterEqual(duration, 0)

        def test_encrypt(self):
            sizes = [128, 256, 512]
            for k in sizes:
                _, params = self.benchmark_method(setup, k)
                _, (pk, sk) = self.benchmark_method(keygen, params)
                x = 
                duration, _ = self.benchmark_method(encrypt, pk, message)
                print(f"Encrypt with size {k} took {duration:.6f} seconds")
                self.assertGreaterEqual(duration, 0)

        def test_decrypt(self):
            sizes = [128, 256, 512]
            for k in sizes:
                _, params = self.benchmark_method(setup, k)
                _, (pk, sk) = self.benchmark_method(keygen, params)
                message = "test message"
                _, ciphertext = self.benchmark_method(encrypt, pk, message)
                duration, _ = self.benchmark_method(decrypt, sk, ciphertext)
                print(f"Decrypt with size {k} took {duration:.6f} seconds")
                self.assertGreaterEqual(duration, 0)

if __name__ == "__main__":
    unittest.main()