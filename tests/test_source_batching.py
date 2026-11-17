"""``source_batch_size`` in ``TDWaveformBase``: a batch generated in chunks of sources must give the
same output as the whole batch at once, for both ``__call__`` (FD and STFT) and
``get_signals_for_residuals``.

The sources have different merger times, so the chunks end up on common grids of different lengths:
this exercises the zero-padding of the STFT segment axis when the chunks are joined.
"""

import sys

# eryn/scienceplots import shims (see tests/test_response_alignment.py)
sys.modules.setdefault("scienceplots", None)
import builtins
import typing

builtins.typing = typing

import unittest

import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import TDSettings
from lisatools.sources.waveformbase import TDPyResponseWaveformBase

_DT = 5.0
_N_ORIG = 20000
_DATA_T0 = 1_000_000.0


class _MiniTDWave(TDPyResponseWaveformBase):
    """Cheap batched sinusoids; the per-source amplitude checks that the arguments are sliced per chunk."""

    def wave_gen_batch(self, amp, ra, dec, merger_time, **kwargs):
        t = np.arange(_N_ORIG, dtype=np.float64) * _DT
        phase = 2.0 * np.pi * 1e-3 * t
        amp = np.asarray(amp)[:, None]
        times = np.broadcast_to(t, (amp.shape[0], _N_ORIG)).copy()
        return times, amp * 1e-21 * np.cos(phase), amp * 1e-21 * np.sin(phase)


class TestSourceBatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orbits = EqualArmlengthOrbits()
        orbits.configure(linear_interp_setup=True)
        cls.orbits = orbits

        rng = np.random.default_rng(1)
        num_bin = 5
        cls.amp = rng.uniform(0.5, 2.0, num_bin)
        cls.ra = rng.uniform(0.0, 2 * np.pi, num_bin)
        cls.dec = rng.uniform(-1.0, 1.0, num_bin)
        # spread over several STFT segments so the chunks get different grid lengths
        cls.merger_time = _DATA_T0 + np.array([0.0, 3.0e4, 1.5e5, 6.0e4, 2.5e5])

    def _make(self, source_batch_size, stft_dt):
        return _MiniTDWave(
            waveform_t0=0.0,
            data_td_settings=TDSettings(N=200000, dt=_DT, t0=_DATA_T0),
            tdi_generation="1st generation",
            tdi_channels="XYZ",
            sampling_frequency=1.0 / _DT,
            orbits=self.orbits,
            order=25,
            buffer_time=10000.0,
            stft_dt=stft_dt,
            freq_min=1e-4,
            freq_max=0.09,
            force_backend="cpu",
            source_batch_size=source_batch_size,
        )

    def _call(self, wf):
        return wf(self.amp, ra=self.ra, dec=self.dec, merger_time=self.merger_time)

    def _check_call(self, stft_dt):
        ref = self._call(self._make(None, stft_dt))
        out = self._call(self._make(2, stft_dt))

        self.assertEqual(len(out), len(ref))
        for a, b in zip(out, ref):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_call_stft(self):
        self._check_call(stft_dt=5000.0)

    def test_call_fd(self):
        self._check_call(stft_dt=None)

    def test_signals_for_residuals(self):
        ref = self._make(None, 5000.0).get_signals_for_residuals(
            self.amp, ra=self.ra, dec=self.dec, merger_time=self.merger_time
        )
        out = self._make(2, 5000.0).get_signals_for_residuals(
            self.amp, ra=self.ra, dec=self.dec, merger_time=self.merger_time
        )

        self.assertEqual(len(out.signals), len(ref.signals))
        for a, b in zip(out.signals, ref.signals):
            np.testing.assert_array_equal(np.asarray(a.arr), np.asarray(b.arr))

    def test_no_split_when_batch_fits(self):
        wf = self._make(10, 5000.0)
        self.assertIsNone(wf._source_chunks((self.amp,), dict(ra=self.ra, dec=self.dec, merger_time=self.merger_time)))


if __name__ == "__main__":
    unittest.main()
