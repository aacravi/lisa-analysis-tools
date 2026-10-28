"""Binary black hole waveform classes (frequency-domain SNR waveforms, PhenomTHM and SEOBNRv5PHM TDI waveforms)."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from typing import Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from bbhx.waveformbuild import BBHWaveformFD

# stft_tof merge: fastlisaresponse is retired -- ResponseWrapper lives in
# lisatools.response.directresponse since Phase 3B.
from lisatools.response.directresponse import ResponseWrapper

from ...domains import DomainSettingsBase
from ...utils.constants import *
from ...utils.typing import NDArrayLike
from ...jax.jaxbase import JaxBase
from ..waveformbase import SNRWaveform, TDPyResponseWaveformBase, TDTDIOnFlyWaveformBase

try:
    import jax.numpy as jnp
    import phentax

    phentax_available = True
except (ImportError, ModuleNotFoundError):
    phentax_available = False
    jnp = np  # type: ignore

# pyseobnr_x is imported lazily in :class:`SEOBNRWaveformBase` (its import is slow).
pyseobnr_available = importlib.util.find_spec("pyseobnr_x") is not None

if pyseobnr_available:
    from pyseobnr_x import generate_waveform as seob_generate_waveform
    from pyseobnr_x.eob.utils.batch_containers import RaggedBatch

if TYPE_CHECKING:
    try:
        import cupy as cp
        from jax import Device
    except (ImportError, ModuleNotFoundError):
        import numpy as cp  # type: ignore
        Device = Any  # type: ignore
        from ...utils.typing import NDArrayLike, ArrayModule


class BBHSNRWaveform(SNRWaveform):
    """Wrapper class for straightforward BBH SNR calculations.

    Calculates it for A and E channels in TDI2.

    Args:
        bbh_waveform_kwargs: ``amp_phase_kwargs`` for :class:`BBHWaveformFD`.
        response_kwargs: ``response_kwargs`` for :class:`BBHWaveformFD`.

    """

    def __init__(
        self,
        bbh_waveform_kwargs: Optional[dict] = {"run_phenomd": False},
        response_kwargs: Optional[dict] = {"TDItag": "AET", "tdi2": True},
    ) -> None:

        if "TDItag" not in response_kwargs:
            response_kwargs["TDItag"] = "AET"

        if "tdi2" not in response_kwargs:
            response_kwargs["tdi2"] = True

        # wave generating class
        self.wave_gen = BBHWaveformFD(
            amp_phase_kwargs=bbh_waveform_kwargs,
            response_kwargs=response_kwargs,
        )

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array."""
        return self._f_arr

    @f_arr.setter
    def f_arr(self, f_arr: np.ndarray) -> None:
        """Set frequency array."""
        self._f_arr = f_arr

    def __call__(
        self,
        *params: Any,
        return_array: Optional[bool] = False,
        mf_min: Optional[float] = 1e-4,
        mf_max: Optional[float] = 0.6,
        freqs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Generate waveforms for SNR calculations.

        Args:
            *params: Parameters for the ``__call__`` function
                for :class:`BBHWaveformFD`.
            return_array: If ``True``, return ``array([A, E, T]).
                If ``False``, return (A, E, T).
            mf_min: Minimum dimensionless frequency to evaluate.
            mf_max: Maximum dimensionless frequency to evaluate.
            freqs: If ``None``, then default will be ``np.logspace(mf_min / M, mf_max / M, 1024)``.
                Otherwise, it will calulate frequencies based on this exact array.
            **kwargs: ``kwargs`` for the ``__call__`` function
                for :class:`BBHWaveformFD`.

        Returns:
            Output waveform.

        """

        # determine frequency array (sparse, log-spaced)
        m1 = params[0]
        m2 = params[1]

        if freqs is None:
            min_f = mf_min / (MTSUN_SI * (m1 + m2))
            max_f = mf_max / (MTSUN_SI * (m1 + m2))
            self.f_arr = np.logspace(np.log10(min_f), np.log10(max_f), 1024)

        else:
            assert isinstance(freqs, np.ndarray)
            self.f_arr = freqs

        # generate waveform with proper settings
        AET = self.wave_gen(
            *params,
            direct=True,
            combine=True,
            freqs=self.f_arr,
            **kwargs,
        )[0]

        # prepare output
        if return_array:
            return AET
        else:
            return (AET[0], AET[1], AET[2])

class BBHTDWaveformMixin:
    """
    Waveform-agnostic helpers for batched time-domain BBH polarizations, shared by :class:`PhenomTHMWaveformBase` and :class:`SEOBNRWaveformBase`.

    Both expect the batch on a common time grid, with each source zero-padded before its start and a mask flagging the samples from the start of each source onwards.
    """

    @staticmethod
    def trim_and_shift_times(
        times: NDArrayLike,
        mask: NDArrayLike,
        *,
        xp: ArrayModule,
        dt: float,
    ) -> NDArrayLike:
        """
        Shift and trim the time arrays for each source according to its mask, so that the resulting time arrays have the same shape (Nbatch, max_valid_times). the initial time points can be different across sources.

        Args:
            times (Array): Time arrays for each source, shape (Nbatch, Ntimes).
            mask (Array): Boolean mask indicating valid time samples for each source, shape (Nbatch, Ntimes).
            xp: Array module (numpy or cupy).
            dt: Time step in seconds.

        Returns:
            Shifted time arrays, shape (Nbatch, max_valid_times).
        """
        valid_points = mask.sum(axis=1)
        max_valid_points = int(valid_points.max())

        times_out = times[:, -max_valid_points:]

        n_pad = max_valid_points - valid_points  # (Nbatch,)

        j = xp.arange(max_valid_points)  # (max_valid_points,)

        is_invalid = j[None, :] < n_pad[:, None]  # (Nbatch, max_valid_points)

        batch_idx = xp.arange(times_out.shape[0])
        first_valid_time = times_out[batch_idx, n_pad]  # (Nbatch,)

        replacement_times = first_valid_time[:, None] - (n_pad[:, None] - j[None, :]) * dt

        times_out = xp.where(is_invalid, replacement_times, times_out)

        return times_out  # shape (Nbatch, max_valid_points)

    @staticmethod
    def _leading_onset_ramp(num_points: int, num_pad: int, taper_length: int, xp: ArrayModule) -> NDArrayLike:
        """
        Get a leading onset ramp to smoothly turn on the waveform.
        This avoids the sharp jump from zero to the first valid point.

        Args:
            num_points: Total number of points in the output array.
            num_pad: Number of initial points that are invalid.
            taper_length: Length of the taper in number of points.
            xp: Array module (numpy or cupy).

        Returns:
            Ramp array of shape (num_points,).
        """
        rj = xp.arange(num_points)
        offset = rj[None, :] - xp.asarray(num_pad, dtype=int)[:, None]
        x = xp.clip(offset / float(taper_length), 0.0, 1.0)
        return 0.5 * (1.0 - xp.cos(xp.pi * x))


class PhenomTHMWaveformBase(JaxBase, BBHTDWaveformMixin):
    """
    Base class for PhenomTHM waveforms.

        This class is not meant to be used directly, but it contains the common code for both :class:`PhenomTHMTDIWaveform` and :class:`PhenomTHMTDIOnFlyWaveform`. In particular, it contains the common code for handling the waveform generation and the reference and starting frequencies.
    
        Args:
            waveform_kwargs: Keyword arguments forwarded to :class:`phentax.waveform.IMRPhenomTHM`.
            Tobs: Observation time in years.
            start_freq: Starting frequency in Hz for the waveform generation. If `None`, it has to be explicitly provided in the waveform generation calls.
            ref_freq: Reference frequency in Hz for the waveform generation. If `None` and `start_freq` is provided, it will default to `start_freq`. Otherwise, it has to be explicitly provided in the waveform generation calls.
    """
    def __init__(
        self,
        waveform_kwargs: dict,
        Tobs: float,
        start_freq: float = None,
        ref_freq: float = None,
        use_reference_time: bool = True,
        use_coalescence_time: bool = True,
    ) -> None:
        
        JaxBase.__init__(self)

        if not phentax_available:
            raise ImportError(
                "PhenomTHM is not available. Please install phentax to use this waveform."
            )

        self.waveform = phentax.waveform.IMRPhenomTHM(T=Tobs, **waveform_kwargs)

        self.start_freq = jnp.asarray(start_freq) if start_freq is not None else None
        self.ref_freq = jnp.asarray(ref_freq) if ref_freq is not None else None
        self.use_reference_time = use_reference_time
        self.ref_time = jnp.asarray(0.0)  if use_coalescence_time else None

    @property
    def phenom_kwargs(self) -> dict:
        """Keyword arguments for the waveform generationDictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""

        waveform_kwargs = dict(
            higher_modes=self.waveform.higher_modes,
            include_negative_modes=self.waveform.include_negative_modes,
            coarse_grain=self.waveform.coarse_grain,
            t_low_fit=self.waveform.t_low == 0.0,
            atol=self.waveform.atol,
            rtol=self.waveform.rtol,
        )

        return {
            "waveform_kwargs": waveform_kwargs,
            "Tobs": self.waveform.T,
            "start_freq": self.start_freq,
            "ref_freq": self.ref_freq,
            "use_reference_time": self.use_reference_time,
            "use_coalescence_time": self.ref_time is not None,
            }

    def get_reference_quantities(self,
                                merger_time: float | NDArrayLike, 
                                start_freq: float | NDArrayLike = None,
                                ref_freq: float | NDArrayLike = None
                                ) -> dict:
        """
        Get the reference quantities for the waveform generation, depending on the settings of the class.

        Args:
            merger_time: Merger time in seconds, shape (Nbatch,).
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.

        Returns:
            Dictionary of reference quantities for the waveform generation.
        """

        start_freq = self._to_jax(start_freq) if start_freq is not None else self.start_freq
        ref_freq = self._to_jax(ref_freq) if ref_freq is not None else self.ref_freq

        if self.use_reference_time:
            ref_time = self._to_jax(-merger_time) if self.ref_time is None else self.ref_time
            return {'t_ref': ref_time, 'f_min': start_freq}
        else:
            return {'f_min': start_freq, 'f_ref': ref_freq}

class PhenomTHMTDIWaveform(TDPyResponseWaveformBase, PhenomTHMWaveformBase):
    """
    Generate PhenomTHM waveforms with the TDI LISA Response.

    Args:
        waveform_kwargs: Keyword arguments forwarded to :class:`phentax.waveform.IMRPhenomTHM`.
        Tobs: Observation time in years.
        start_freq: Starting frequency in Hz for the waveform generation. If `None`, it has to be explicitly provided in the waveform generation calls.
        ref_freq: Reference frequency in Hz for the waveform generation. If `None` and `start_freq` is provided, it will default to `start_freq`. Otherwise, it has to be explicitly provided in the waveform generation calls.
        *args: Additional positional arguments forwarded to :class:`TDPyResponseWaveformBase`.
        **kwargs: Additional keyword arguments forwarded to :class:`TDPyResponseWaveformBase`.
    """

    def __init__(
        self,
        waveform_kwargs: dict,
        Tobs: float = 1.0,
        start_freq: float = None,
        ref_freq: float = None,
        use_reference_time: bool = True,
        use_coalescence_time: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        if "coarse_grain" in waveform_kwargs and waveform_kwargs["coarse_grain"] is True:
            raise ValueError("Applying the response through the `PyresponseTDI` class requires equispaced time arrays")

        TDPyResponseWaveformBase.__init__(
            self,
            *args,
            **kwargs,
        )
        
        PhenomTHMWaveformBase.__init__(
            self,
            waveform_kwargs=waveform_kwargs,
            Tobs=Tobs,
            start_freq=start_freq,
            ref_freq=ref_freq,
            use_reference_time=use_reference_time,
            use_coalescence_time=use_coalescence_time,
        )

    @property
    def kwargs(self) -> dict:
        """Dictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""

        phenom_kwargs = self.phenom_kwargs  # get the PhenomTHM-specific kwargs
        wrapper_kwargs = self.wrapper_kwargs # get the TD TDI wrapper-specific kwargs

        return {
            **phenom_kwargs,
            **wrapper_kwargs,
        }
        
    def wave_gen(
        self,
        m1: float,
        m2: float,
        s1z: float,
        s2z: float,
        distance: float,
        phi_ref: float,
        inclination: float,
        psi: float,
        ra: float = None,
        dec: float = None,
        merger_time: float = None,
        ref_freq: float = None,
        start_freq: float = None,
        synchronize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the waveform's polarizations for a single source.

        Args:
            m1: Source-1 mass in solar masses.
            m2: Source-2 mass in solar masses.
            s1z: Dimensionless spin of source 1.
            s2z: Dimensionless spin of source 2.
            distance: Luminosity distance in Mpc.
            phi_ref: Reference phase in radians.
            inclination: Inclination angle in radians.
            psi: Polarisation angle in radians.
            ra: Right ascension for the source in radians.
            dec: Declination for the source in radians.
            merger_time: Time of merger in seconds.
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.
            synchronize: If `True`, it will call `block_until_ready()` on the waveform outputs. :meth:`self._from_jax` relies on `dlpack` to move data from JAX to Cupy, that should already handle CUDA streams correctly.

        Returns:
            t_arr, h_plus, h_cross

        """

        reference_kwargs = self.get_reference_quantities(merger_time=merger_time, start_freq=start_freq, ref_freq=ref_freq)

        times, mask, hplus, hcross = self.waveform.compute_polarizations_at_once(
            self._to_jax(m1),
            self._to_jax(m2),
            self._to_jax(s1z),
            self._to_jax(s2z),
            self._to_jax(distance),
            self._to_jax(phi_ref),
            self._to_jax(inclination),
            self._to_jax(psi),
            delta_t=self.dt,
            **reference_kwargs,
        )

        xp_mask = self._from_jax(mask, do_synchronize=synchronize)
        out_times = self._from_jax(times, do_synchronize=synchronize)[xp_mask]
        out_hplus = self._from_jax(hplus, do_synchronize=synchronize)[xp_mask]
        out_hcross = self._from_jax(hcross, do_synchronize=synchronize)[xp_mask]        

        return (
            out_times,
            out_hplus,
            out_hcross,
        )

    def wave_gen_batch(
        self,
        m1: NDArrayLike,
        m2: NDArrayLike,
        s1z: NDArrayLike,
        s2z: NDArrayLike,
        distance: NDArrayLike,
        phi_ref: NDArrayLike,
        inclination: NDArrayLike,
        psi: NDArrayLike,
        ra: NDArrayLike = None,
        dec: NDArrayLike = None,
        merger_time: NDArrayLike = None,
        ref_freq: float | NDArrayLike = None,
        start_freq: float | NDArrayLike = None,
        synchronize: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate polarizations for a batch of sources using phentax's vectorised path.

        phentax uses JAX vmap internally, so all parameters must be broadcastable
        to the batch shape.  The returned arrays still carry the padded time axis;
        per-source masking is handled by the caller (:meth:`_call_batched`).

        Args:
            m1: Source-1 masses in solar masses, shape (Nbatch,).
            m2: Source-2 masses in solar masses, shape (Nbatch,).
            s1z: Dimensionless spin of source 1, shape (Nbatch,).
            s2z: Dimensionless spin of source 2, shape (Nbatch,).
            distance: Luminosity distance in Mpc, shape (Nbatch,).
            phi_ref: Reference phase in radians, shape (Nbatch,).
            inclination: Inclination angle in radians, shape (Nbatch,).
            psi: Polarisation angle in radians, shape (Nbatch,).
            ref_freq: Reference frequency in Hz, float.
            start_freq: Starting frequency in Hz, float.
            ra: Right ascension in radians, shape (Nbatch,).
            dec: Declination in radians, shape (Nbatch,).
            merger_time: Time of merger in seconds, shape (Nbatch,).
            synchronize: If `True`, it will call `block_until_ready()` on the waveform outputs. :meth:`self._from_jax` relies on `dlpack` to move data from JAX to Cupy, that should already handle CUDA streams correctly.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_polarizations_at_once`` (e.g. ``T`` for observation time
                override, ``t_min``, ``t_ref``).

        Returns:
            Tuple of (times_batch, h_plus_batch, h_cross_batch),
            each of shape (Nbatch, N_valid_times) as plain NumPy or Cupy arrays.
        """

        reference_kwargs = self.get_reference_quantities(merger_time=merger_time, start_freq=start_freq, ref_freq=ref_freq)

        times, mask, hplus, hcross = self.waveform.compute_polarizations_at_once(
            self._to_jax(m1),
            self._to_jax(m2),
            self._to_jax(s1z),
            self._to_jax(s2z),
            self._to_jax(distance),
            self._to_jax(phi_ref),
            self._to_jax(inclination),
            self._to_jax(psi),
            delta_t=self.dt,
            **reference_kwargs,
            **kwargs,
        )
        
        times = self._from_jax(times, do_synchronize=synchronize)
        mask = self._from_jax(mask, do_synchronize=synchronize)
        hplus = self._from_jax(hplus, do_synchronize=synchronize)
        hcross = self._from_jax(hcross, do_synchronize=synchronize)

        times_out = self.trim_and_shift_times(times, mask, xp=self.xp, dt=self.dt)
        num_keep = times_out.shape[-1]

        num_pad = num_keep - mask.sum(axis=1).astype(int)  # (Nbatch,)
        taper_length = int(self.tdi_buffer_time * 5 / self.dt)
        ramp = self._leading_onset_ramp(num_points=num_keep, num_pad=num_pad, taper_length=taper_length, xp=self.xp)  # (Nbatch, num_keep)

        return (
            times_out,
            hplus[:, -num_keep:] * ramp,  # apply ramp to h_plus to avoid sharp jump at the beginning of the waveform, which can cause issues with the TDI response
            hcross[:, -num_keep:] * ramp,  # apply ramp to h_cross to avoid sharp jump at the beginning of the waveform, which can cause issues with the TDI response
        )


class PhenomTHMTDIOnFlyWaveform(TDTDIOnFlyWaveformBase, PhenomTHMWaveformBase):
    """
    Generate PhenomTHM waveforms with the TDI LISA Response, on the fly.

    This class inherits from :class:`TDTDIOnFlyWaveformBase`, which means that it generates the waveform sparsely and evaluate the tdi delays on splined versions of each mode's amplitude and phase.

    Args:

    """

    def __init__(
        self,
        waveform_kwargs: dict,
        Tobs: float,
        start_freq: float = None,
        ref_freq: float = None,
        use_reference_time: bool = True,
        use_coalescence_time: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        TDTDIOnFlyWaveformBase.__init__(
            self,
            *args,
            **kwargs,
        )

        PhenomTHMWaveformBase.__init__(
            self,
            waveform_kwargs=waveform_kwargs,
            Tobs=Tobs,
            start_freq=start_freq,
            ref_freq=ref_freq,
            use_reference_time=use_reference_time,
            use_coalescence_time=use_coalescence_time,
        )

    @property
    def kwargs(self) -> dict:
        """Dictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""

        phenom_kwargs = self.phenom_kwargs  # get the PhenomTHM-specific kwargs
        wrapper_kwargs = self.wrapper_kwargs # get the TD TDI wrapper-specific kwargs

        return {
            **phenom_kwargs,
            **wrapper_kwargs,
        }
        
    def get_amp_phase(
        self,
        m1: NDArrayLike,
        m2: NDArrayLike,
        s1z: NDArrayLike,
        s2z: NDArrayLike,
        distance: NDArrayLike,
        phi_ref: NDArrayLike,
        inclination: NDArrayLike,
        psi: NDArrayLike,
        ra: NDArrayLike = None,
        dec: NDArrayLike = None,
        merger_time: NDArrayLike = None,
        ref_freq: NDArrayLike = None,
        start_freq: NDArrayLike = None,
        synchronize: bool = False,
        **kwargs,
    ) -> Tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
        """
        Get the waveform modes' amplitude and phase, already including the spherical harmonic contributions.

        Args:
            m1: Source-1 masses in solar masses, shape (Nbatch,).
            m2: Source-2 masses in solar masses, shape (Nbatch,).
            s1z: Dimensionless spin of source 1, shape (Nbatch,).
            s2z: Dimensionless spin of source 2, shape (Nbatch,).
            distance: Luminosity distance in Mpc, shape (Nbatch,).
            phi_ref: Reference phase in radians, shape (Nbatch,).
            inclination: Inclination angle in radians, shape (Nbatch,).
            psi: Polarisation angle in radians, shape (Nbatch,).
            ra: Right ascension for the sources in radians, shape (Nbatch,).
            dec: Declination for the sources in radians, shape (Nbatch,).
            merger_time: Time of merger in seconds, shape (Nbatch,).
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.
            synchronize: If `True`, it will call `block_until_ready()` on the waveform outputs. :meth:`self._from_jax` relies on `dlpack` to move data from JAX to Cupy, that should already handle CUDA streams correctly.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_strain_components_amp_phase``.

        Returns:
            times, mode amplitudes and mode phases
        """
        
        reference_kwargs = self.get_reference_quantities(merger_time=merger_time, start_freq=start_freq, ref_freq=ref_freq)

        times, mask, amplitude, phase = self.waveform.compute_strain_components_amp_phase(
            self._to_jax(m1),
            self._to_jax(m2),
            self._to_jax(s1z),
            self._to_jax(s2z),
            self._to_jax(distance),
            self._to_jax(phi_ref),
            self._to_jax(inclination),
            self._to_jax(psi),
            delta_t=self.dt,
            **reference_kwargs
        )
        #amplitude.block_until_ready()  # ensure all outputs are ready before moving to self.xp

        times = self._from_jax(times, do_synchronize=synchronize)
        mask = self._from_jax(mask, do_synchronize=synchronize)
        amplitude = self._from_jax(amplitude, do_synchronize=synchronize)
        phase = self._from_jax(phase, do_synchronize=synchronize)

        times_out = self.trim_and_shift_times(times, mask, xp=self.xp, dt=self.dt)

        num_keep = times_out.shape[-1]

        num_pad = num_keep - mask.sum(axis=1).astype(int)  # (Nbatch,)
        taper_length = int(self.tdi_buffer_time * 5 / self.dt)
        ramp = self._leading_onset_ramp(num_points=num_keep, num_pad=num_pad, taper_length=taper_length, xp=self.xp)  # (Nbatch, num_keep)

        return times_out, amplitude[..., -num_keep:] * ramp, phase[..., -num_keep:] * ramp # should we also apply the ramp to the phase? 

    def process_amp_phase(
        self, amp: NDArrayLike, phase: NDArrayLike
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """
        Process the waveform amplitude and phase to align to the tdi on fly conventions.
        For this specific waveform, we need :math:`A \\to A/2` and :math:`\\phi \\to \\pi-\\phi`.

        Args:
            amp (Array): Waveform modes amplitude, already including the spherical harmonic contributions. Shape (num_binaries, num_modes, num_times).
            phase (Array): Waveform modes phase, already including the spherical harmonic contributions. Shape (num_binaries, num_modes, num_times).

        Returns:
            Tuple of (processed_amplitude, processed_phase), each of shape (num_binaries, num_modes, num_times).
        """

        return (amp / 2.0, self.xp.pi - phase)

class SEOBNRWaveformBase(BBHTDWaveformMixin):
    """
    Base class for SEOBNRv5PHM waveforms generated with ``pyseobnr_x``.

    This class is not meant to be used directly, but it contains the common code for :class:`SEOBNRTDIWaveform`.
    It mirrors :class:`PhenomTHMWaveformBase`: the batch of binaries is evaluated in a single ``pyseobnr_x``
    call, and the polarizations of every binary are retrieved in a single kernel call on a common time grid
    with spacing ``self.dt`` and :math:`t=0` at the peak of the co-precessing (2,2) amplitude (``LISA`` context).

    Args:
        waveform_kwargs: Computation parameters forwarded to :class:`pyseobnr_x.generate_waveform.GenerateWaveform`
            (e.g. ``ModeArray``, ``postadiabatic``, ``ode_backend``, ``frame``...). They override the defaults
            (``context="LISA"`` plus the CPU or GPU switches, chosen from ``force_backend``).
        Tobs: Maximum length of the waveform in seconds. Longer waveforms are cut at the start.
        start_freq: Starting frequency of the (2,2) mode in Hz. If `None`, it has to be explicitly provided in the waveform generation calls.
        ref_freq: Reference frequency in Hz. If `None`, it defaults to the starting frequency.
    """

    CPU_SWITCHES = dict(
        ode_backend="boost",
        pn_ode_backend="boost",
        hamiltonian="generated",
        rholm="generated",
        frame="cpp",
        inspiral_modes="cpp",
        antisymmetric_stage="cpp",
    )

    GPU_SWITCHES = dict(
        CPU_SWITCHES,
        frame="cuda",
        inspiral_modes="cuda",
        antisymmetric_stage="cuda",
    )

    def __init__(
        self,
        waveform_kwargs: dict,
        Tobs: float,
        start_freq: float = None,
        ref_freq: float = None,
    ) -> None:

        if not pyseobnr_available:
            raise ImportError(
                "SEOBNRv5PHM is not available. Please install pyseobnr_x to use this waveform."
            )

        self._seob = seob_generate_waveform
        self._RaggedBatch = RaggedBatch

        use_gpu = self.force_backend is not None and self.force_backend != "cpu"
        switches = self.GPU_SWITCHES if use_gpu else self.CPU_SWITCHES

        self.waveform_kwargs = dict(waveform_kwargs)
        self.computation_parameters = {"context": "LISA", **switches, **self.waveform_kwargs}

        self.T_obs = Tobs
        self.start_freq = start_freq
        self.ref_freq = ref_freq

    @property
    def seob_kwargs(self) -> dict:
        """Dictionary of SEOBNR settings used to initialize the waveform, for reproducibility and debugging."""

        return {
            "waveform_kwargs": self.waveform_kwargs,
            "Tobs": self.T_obs,
            "start_freq": self.start_freq,
            "ref_freq": self.ref_freq,
        }

    def compute_polarizations_at_once(
        self,
        m1: NDArrayLike,
        m2: NDArrayLike,
        s1z: NDArrayLike,
        s2z: NDArrayLike,
        distance: NDArrayLike,
        phi_ref: NDArrayLike,
        inclination: NDArrayLike,
        psi: NDArrayLike,
        start_freq: float | NDArrayLike = None,
        ref_freq: float | NDArrayLike = None,
    ) -> Tuple[NDArrayLike, NDArrayLike, NDArrayLike, NDArrayLike]:
        """
        Generate the polarizations of a batch of sources on a common time grid.

        Same output layout as :meth:`phentax.waveform.IMRPhenomTHM.compute_polarizations_at_once`: each source is
        zero outside of its own valid time range, and ``mask`` flags the samples from the start of each waveform
        to the end of the grid (so that the valid samples are right-aligned, as :meth:`trim_and_shift_times` expects).

        Args:
            m1: Source-1 masses in solar masses, shape (Nbatch,).
            m2: Source-2 masses in solar masses, shape (Nbatch,).
            s1z: Dimensionless spin of source 1, shape (Nbatch,).
            s2z: Dimensionless spin of source 2, shape (Nbatch,).
            distance: Luminosity distance in Mpc, shape (Nbatch,).
            phi_ref: Reference phase in radians, shape (Nbatch,).
            inclination: Inclination angle in radians, shape (Nbatch,).
            psi: Polarisation angle in radians, shape (Nbatch,).
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq`.
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq`, then to the starting frequency.

        Returns:
            times (Ntimes,), mask (Nbatch, Ntimes), h_plus (Nbatch, Ntimes), h_cross (Nbatch, Ntimes), on ``self.xp``.
        """

        start_freq = start_freq if start_freq is not None else self.start_freq
        ref_freq = ref_freq if ref_freq is not None else self.ref_freq
        if start_freq is None:
            raise ValueError("start_freq has to be provided, either at initialization or in the call.")
        if ref_freq is None:
            ref_freq = start_freq

        # pyseobnr_x works on the host
        to_host = lambda a: np.atleast_1d(a.get() if hasattr(a, "get") else np.asarray(a)).astype(np.float64)
        m1, m2, s1z, s2z, distance, phi_ref, inclination, psi = (
            to_host(arg) for arg in (m1, m2, s1z, s2z, distance, phi_ref, inclination, psi)
        )
        num_bin = m1.shape[0]

        binaries = pd.DataFrame(
            dict(
                mass1=m1,
                mass2=m2,
                spin1z=s1z,
                spin2z=s2z,
                distance=distance,
                phi_ref=phi_ref,
                inclination=inclination,
                f22_start=np.broadcast_to(to_host(start_freq), (num_bin,)),
                f_ref=np.broadcast_to(to_host(ref_freq), (num_bin,)),
                deltaT=self.dt,
            )
        )

        generator = self._seob.GenerateWaveform(binaries, self.computation_parameters)
        generator.evaluate_model()
        model = generator.model

        if not model.evaluated.all():
            failed = np.flatnonzero(~model.evaluated)
            raise RuntimeError(
                f"SEOBNRv5PHM evaluation failed for sources {failed.tolist()}: "
                f"{ {int(i): str(model.errors.get(i)) for i in failed} }"
            )

        # valid range of each waveform, with t=0 at the (2,2) amplitude peak
        output = model.batch_output
        mtot = np.array([p["mass1"] + p["mass2"] for p in generator.parameters_list])
        dist = np.array([p["distance"] for p in generator.parameters_list])
        mtot_sec = mtot * self._seob.lal.MTSUN_SI

        t_shift = np.asarray(output.t_shift[:num_bin])
        t_lo_geo = np.asarray(output.t_start[:num_bin]) - t_shift
        t_hi_geo = np.asarray(output.t_duration[:num_bin]) - t_shift

        t_hi = t_hi_geo * mtot_sec
        t_lo = np.maximum(t_lo_geo * mtot_sec, t_hi - self.T_obs)

        # common grid on integer multiples of dt, so the merger sits on the grid
        k_lo = int(np.ceil(t_lo.min() / self.dt))
        k_hi = int(np.floor(t_hi.max() / self.dt))
        times = np.arange(k_lo, k_hi + 1) * self.dt

        # per-source geometric time samples, retrieved in one kernel call
        sel = (times[None, :] >= t_lo[:, None]) & (times[None, :] <= t_hi[:, None])
        times_geo = times[None, :] / mtot_sec[:, None]
        sel &= (times_geo >= t_lo_geo[:, None]) & (times_geo <= t_hi_geo[:, None])

        counts = sel.sum(axis=1)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        rows = model.polarizations.evaluate(
            self._RaggedBatch(values=np.ascontiguousarray(times_geo[sel]), offsets=offsets),
            np.arange(num_bin, dtype=np.intp),
            time_shift=t_shift,
        )

        # h+ - i hx, with the distance and LAL sign convention of GenerateWaveform
        fac = -mtot * self._seob.lal.MRSUN_SI / (dist * self._seob.Mpc_to_meters)
        hpc = np.zeros(sel.shape, dtype=np.complex128)
        hpc[sel] = rows * np.repeat(fac, counts)

        # rotate by the polarization angle, as phentax does
        cos_2psi = np.cos(2.0 * psi)[:, None]
        sin_2psi = np.sin(2.0 * psi)[:, None]
        hplus = hpc.real * cos_2psi + hpc.imag * sin_2psi
        hcross = hpc.real * sin_2psi - hpc.imag * cos_2psi

        mask = times[None, :] >= times[sel.argmax(axis=1)][:, None]

        return (
            self.xp.asarray(times),
            self.xp.asarray(mask),
            self.xp.asarray(hplus),
            self.xp.asarray(hcross),
        )


class SEOBNRTDIWaveform(TDPyResponseWaveformBase, SEOBNRWaveformBase):
    """
    Generate SEOBNRv5PHM waveforms with the TDI LISA Response.

    Drop-in replacement for :class:`PhenomTHMTDIWaveform` (same parameters, same outputs), with aligned spins.

    Args:
        waveform_kwargs: Computation parameters forwarded to :class:`pyseobnr_x.generate_waveform.GenerateWaveform`.
        Tobs: Maximum length of the waveform in seconds.
        start_freq: Starting frequency in Hz for the waveform generation. If `None`, it has to be explicitly provided in the waveform generation calls.
        ref_freq: Reference frequency in Hz for the waveform generation. If `None`, it defaults to the starting frequency.
        *args: Additional positional arguments forwarded to :class:`TDPyResponseWaveformBase`.
        **kwargs: Additional keyword arguments forwarded to :class:`TDPyResponseWaveformBase`.
    """

    def __init__(
        self,
        waveform_kwargs: dict,
        Tobs: float = 1.0,
        start_freq: float = None,
        ref_freq: float = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        TDPyResponseWaveformBase.__init__(
            self,
            *args,
            **kwargs,
        )

        SEOBNRWaveformBase.__init__(
            self,
            waveform_kwargs=waveform_kwargs,
            Tobs=Tobs,
            start_freq=start_freq,
            ref_freq=ref_freq,
        )

    @property
    def kwargs(self) -> dict:
        """Dictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""

        seob_kwargs = self.seob_kwargs  # get the SEOBNR-specific kwargs
        wrapper_kwargs = self.wrapper_kwargs # get the TD TDI wrapper-specific kwargs

        return {
            **seob_kwargs,
            **wrapper_kwargs,
        }

    def wave_gen(
        self,
        m1: float,
        m2: float,
        s1z: float,
        s2z: float,
        distance: float,
        phi_ref: float,
        inclination: float,
        psi: float,
        ra: float = None,
        dec: float = None,
        merger_time: float = None,
        ref_freq: float = None,
        start_freq: float = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the waveform's polarizations for a single source.

        Args:
            m1: Source-1 mass in solar masses.
            m2: Source-2 mass in solar masses.
            s1z: Dimensionless spin of source 1.
            s2z: Dimensionless spin of source 2.
            distance: Luminosity distance in Mpc.
            phi_ref: Reference phase in radians.
            inclination: Inclination angle in radians.
            psi: Polarisation angle in radians.
            ra: Right ascension for the source in radians.
            dec: Declination for the source in radians.
            merger_time: Time of merger in seconds.
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq`, then to the starting frequency.
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq`.

        Returns:
            t_arr, h_plus, h_cross

        """

        times, mask, hplus, hcross = self.compute_polarizations_at_once(
            m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
            start_freq=start_freq,
            ref_freq=ref_freq,
        )

        return (
            times[mask[0]],
            hplus[0][mask[0]],
            hcross[0][mask[0]],
        )

    def wave_gen_batch(
        self,
        m1: NDArrayLike,
        m2: NDArrayLike,
        s1z: NDArrayLike,
        s2z: NDArrayLike,
        distance: NDArrayLike,
        phi_ref: NDArrayLike,
        inclination: NDArrayLike,
        psi: NDArrayLike,
        ra: NDArrayLike = None,
        dec: NDArrayLike = None,
        merger_time: NDArrayLike = None,
        ref_freq: float | NDArrayLike = None,
        start_freq: float | NDArrayLike = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate polarizations for a batch of sources using pyseobnr_x's batched path.

        Args:
            m1: Source-1 masses in solar masses, shape (Nbatch,).
            m2: Source-2 masses in solar masses, shape (Nbatch,).
            s1z: Dimensionless spin of source 1, shape (Nbatch,).
            s2z: Dimensionless spin of source 2, shape (Nbatch,).
            distance: Luminosity distance in Mpc, shape (Nbatch,).
            phi_ref: Reference phase in radians, shape (Nbatch,).
            inclination: Inclination angle in radians, shape (Nbatch,).
            psi: Polarisation angle in radians, shape (Nbatch,).
            ra: Right ascension in radians, shape (Nbatch,).
            dec: Declination in radians, shape (Nbatch,).
            merger_time: Time of merger in seconds, shape (Nbatch,).
            ref_freq: Reference frequency in Hz, float or shape (Nbatch,).
            start_freq: Starting frequency in Hz, float or shape (Nbatch,).
            **kwargs: ``f_ref`` / ``f_min`` are accepted as aliases of ``ref_freq`` / ``start_freq``
                (phentax naming); anything else is ignored.

        Returns:
            Tuple of (times_batch, h_plus_batch, h_cross_batch),
            each of shape (Nbatch, N_valid_times) as plain NumPy or Cupy arrays.
        """

        ref_freq = kwargs.pop("f_ref", ref_freq)
        start_freq = kwargs.pop("f_min", start_freq)

        times, mask, hplus, hcross = self.compute_polarizations_at_once(
            m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
            start_freq=start_freq,
            ref_freq=ref_freq,
        )
        times = self.xp.broadcast_to(times, mask.shape)

        times_out = self.trim_and_shift_times(times, mask, xp=self.xp, dt=self.dt)
        num_keep = times_out.shape[-1]

        num_pad = num_keep - mask.sum(axis=1).astype(int)  # (Nbatch,)
        taper_length = int(self.tdi_buffer_time * 5 / self.dt)
        ramp = self._leading_onset_ramp(num_points=num_keep, num_pad=num_pad, taper_length=taper_length, xp=self.xp)  # (Nbatch, num_keep)

        return (
            times_out,
            hplus[:, -num_keep:] * ramp,  # apply ramp to h_plus to avoid sharp jump at the beginning of the waveform, which can cause issues with the TDI response
            hcross[:, -num_keep:] * ramp,  # apply ramp to h_cross to avoid sharp jump at the beginning of the waveform, which can cause issues with the TDI response
        )
