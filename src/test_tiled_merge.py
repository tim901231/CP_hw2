"""Numerical regression checks: python -m unittest discover -s src -p test_tiled_merge.py"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from main import merge, optimal_raw_weights
from tiled_merge import merge_tiff_stack, merge_raw_tiff_stack, read_patch


class TiledMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(42)
        self.times = [.01, .04, .2]

    def tearDown(self):
        self.temp.cleanup()

    def write_stack(self, arrays):
        paths = []
        for i, a in enumerate(arrays):
            p = self.root / f'{i}.tiff'
            tifffile.imwrite(p, a)
            paths.append(p)
        return paths

    def test_rgb_all_modes_weights_and_patch_edges(self):
        arrays = self.rng.integers(0, 65536, size=(3, 13, 17, 3), dtype=np.uint16)
        arrays[:, 0, 0] = 0
        arrays[:, 0, 1] = 65535
        paths = self.write_stack(arrays)
        observed = arrays.astype(np.float32) / 65535.
        for mode in ['linear', 'log']:
            for weight in ['uniform', 'tent', 'gaussian', 'photon', 'optimal']:
                if mode == 'log' and weight == 'optimal':
                    continue
                with self.subTest(mode=mode, weight=weight):
                    kwargs = dict(config=mode, weight=weight, exposure_times=self.times,
                                  gain=[.01, .02, .03], additive_variance=[.001]*3)
                    expected = merge(observed, verbose=False, **kwargs)
                    actual = merge_tiff_stack(paths, merge, tile_size=5, progress=False, **kwargs)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, np.float32)

    def test_raw_odd_tiles_bayer_phase_signed_signal_and_fallback(self):
        arrays = self.rng.integers(1, 4096, size=(3, 13, 17), dtype=np.uint16)
        arrays[:, 0, 0] = 0
        arrays[:, 1, 1] = 4095
        arrays[:, 2, 2] = 100
        gains, additive = np.array([.3, .25, .4]), np.array([1., 2., 3.])
        num, den = np.zeros((13, 17)), np.zeros((13, 17))
        fallback = (arrays[0].astype(np.float32) - 150.) / self.times[0]
        for raw, t in zip(arrays, self.times):
            for dy, dx, c in [(0,0,0), (0,1,1), (1,0,1), (1,1,2)]:
                sample = raw[dy::2, dx::2]
                signal = sample.astype(float) - 150.
                w = optimal_raw_weights(signal, t, gains[c], additive[c],
                                        (sample > 0) & (sample < 4000.))
                num[dy::2, dx::2] += w * signal / t
                den[dy::2, dx::2] += w
        valid = den > 0
        np.divide(num, den, out=fallback, where=valid, casting='unsafe')
        paths = self.write_stack(arrays)
        for tile_size in [1, 5, 16, 256]:
            actual, mask = merge_raw_tiff_stack(paths, self.times, gains, additive,
                                               tile_size=tile_size, progress=False)
            np.testing.assert_array_equal(actual, fallback)
            np.testing.assert_array_equal(mask, valid)
            self.assertLess(actual[2, 2], 0)

    def test_reject_invalid_inputs(self):
        paths = self.write_stack(np.ones((3, 7, 9, 3), dtype=np.uint16))
        for size in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                merge_tiff_stack(paths, merge, tile_size=size, progress=False)
        with self.assertRaises(ValueError):
            merge_tiff_stack(paths, merge, exposure_times=[1], progress=False)
        tifffile.imwrite(paths[-1], np.ones((8, 9, 3), dtype=np.uint16))
        with self.assertRaises(ValueError):
            merge_tiff_stack(paths, merge, progress=False)

    def test_compressed_tiff_fails_without_full_image_decode(self):
        path = self.root / 'compressed.tiff'
        tifffile.imwrite(path, np.ones((8, 9), dtype=np.uint16), compression='deflate')
        with self.assertRaisesRegex(ValueError, 'uncompressed'):
            read_patch(path, slice(0, 2), slice(0, 2))


if __name__ == '__main__':
    unittest.main()
