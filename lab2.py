import unittest
import torch
from torch.utils.cpp_extension import load


class LabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext = load(
            name='my_extension',
            sources=['lab2.cu'],
            extra_cuda_cflags=['-O3'],
            extra_cflags=['-O3'],
        )

    def test_mult(self):
        n = torch.randint(size=(1,), low=1, high=2048)

        x = torch.rand((n,), device='cuda')
        y = torch.tensor([127.0], device='cuda')
        z = LabTest.ext.prod(x, y)

        z_ = x * y

        self.assertTrue(torch.allclose(z, z_, atol=1e-7, rtol=1e-6))


unittest.main()
