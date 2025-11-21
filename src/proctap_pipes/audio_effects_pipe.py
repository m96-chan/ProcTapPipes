"""Audio effects pipe for real-time audio processing.

This pipe applies various audio effects (noise reduction, normalization, EQ, etc.)
while passing through processed audio data, allowing it to be used in pipelines.
"""

from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from proctap_pipes.base import AudioFormat, BasePipe


class NoiseReductionMode(str, Enum):
    """Noise reduction algorithm modes."""

    SPECTRAL_GATE = "spectral_gate"  # Spectral gating (moderate speed, good quality)
    WIENER = "wiener"  # Wiener filter (slow, best quality)
    SIMPLE = "simple"  # Simple threshold-based (fast, basic quality)


class AudioEffectsPipe(BasePipe):
    """Real-time audio effects processing pipe with passthrough.

    Applies various audio effects to improve audio quality for transcription
    or other downstream processing tasks.

    Available effects:
    - Noise reduction (spectral gating, Wiener filter)
    - Volume normalization
    - High-pass filter (remove low-frequency rumble)
    - Low-pass filter (remove high-frequency hiss)

    Example:
        # CLI usage
        proctap -pid 1234 --stdout | proctap-effects --denoise | proctap-whisper

        # Python API usage
        pipe = AudioEffectsPipe(denoise=True, normalize=True, highpass=80)
        for processed_chunk in pipe.run_stream(audio_stream):
            # processed_chunk is the enhanced audio data
            downstream_process(processed_chunk)
    """

    def __init__(
        self,
        audio_format: AudioFormat | None = None,
        denoise: bool = False,
        noise_reduction_mode: NoiseReductionMode = NoiseReductionMode.SPECTRAL_GATE,
        noise_threshold: float = 0.02,
        normalize: bool = False,
        target_level: float = 0.7,
        highpass: float | None = None,
        lowpass: float | None = None,
        verbose: bool = False,
    ):
        """Initialize audio effects pipe.

        Args:
            audio_format: Audio format configuration
            denoise: Enable noise reduction
            noise_reduction_mode: Noise reduction algorithm to use
            noise_threshold: Threshold for noise reduction (0.0-1.0, lower = more aggressive)
            normalize: Enable volume normalization
            target_level: Target volume level for normalization (0.0-1.0)
            highpass: High-pass filter cutoff frequency in Hz (removes low frequencies)
            lowpass: Low-pass filter cutoff frequency in Hz (removes high frequencies)
            verbose: Enable verbose logging
        """
        super().__init__(audio_format)
        self.denoise = denoise
        self.noise_reduction_mode = noise_reduction_mode
        self.noise_threshold = noise_threshold
        self.normalize = normalize
        self.target_level = target_level
        self.highpass = highpass
        self.lowpass = lowpass
        self.verbose = verbose

        # State for spectral gate noise reduction
        self.noise_profile: npt.NDArray[Any] | None = None
        self.profile_frames = 0
        self.max_profile_frames = 10  # Use first 10 frames to build noise profile

        if self.verbose:
            effects = []
            if self.denoise:
                effects.append(f"denoise ({self.noise_reduction_mode})")
            if self.normalize:
                effects.append(f"normalize (target={self.target_level})")
            if self.highpass:
                effects.append(f"highpass ({self.highpass}Hz)")
            if self.lowpass:
                effects.append(f"lowpass ({self.lowpass}Hz)")

            if effects:
                self.logger.info(f"Audio effects enabled: {', '.join(effects)}")
            else:
                self.logger.warning("No audio effects enabled")

    def _to_float(self, audio_data: npt.NDArray[Any]) -> npt.NDArray[np.floating[Any]]:
        """Convert audio data to float32 normalized to [-1, 1].

        Args:
            audio_data: Audio samples (any dtype)

        Returns:
            Float32 audio normalized to [-1, 1]
        """
        if audio_data.dtype == np.int16:
            return audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            return audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            return (audio_data.astype(np.float32) - 128) / 128.0
        else:
            return audio_data.astype(np.float32)

    def _from_float(
        self, audio_data: npt.NDArray[np.floating[Any]], target_dtype: npt.DTypeLike
    ) -> npt.NDArray[Any]:
        """Convert float audio back to original dtype.

        Args:
            audio_data: Float32 audio normalized to [-1, 1]
            target_dtype: Target dtype

        Returns:
            Audio in target dtype
        """
        # Clip to prevent overflow
        audio_data = np.clip(audio_data, -1.0, 1.0)

        if target_dtype == np.int16:
            return (audio_data * 32767.0).astype(np.int16)
        elif target_dtype == np.int32:
            return (audio_data * 2147483647.0).astype(np.int32)
        elif target_dtype == np.uint8:
            return ((audio_data * 128.0) + 128).astype(np.uint8)
        else:
            return audio_data.astype(target_dtype)

    def _apply_simple_denoise(
        self, audio_data: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply simple threshold-based noise reduction (fast).

        This method uses a simple threshold to reduce low-amplitude noise.
        Much faster than FFT-based methods but less effective.

        Args:
            audio_data: Float audio data, shape (samples, channels)

        Returns:
            Denoised audio data
        """
        # Calculate noise threshold based on the first few frames
        if self.noise_profile is None or self.profile_frames < self.max_profile_frames:
            # Build noise profile from RMS of current chunk
            rms = np.sqrt(np.mean(audio_data**2))
            if self.noise_profile is None:
                self.noise_profile = np.array([rms])
            else:
                self.noise_profile = (self.noise_profile + rms) / 2.0
            self.profile_frames += 1

        # Apply threshold-based noise reduction
        if self.noise_profile is not None:
            threshold = float(self.noise_profile[0]) * (1.0 + self.noise_threshold * 10)

            # Apply soft threshold (gradual reduction)
            amplitude = np.abs(audio_data)
            gain = np.where(
                amplitude > threshold,
                1.0,
                amplitude / (threshold + 1e-10),  # Gradual reduction below threshold
            )

            return audio_data * gain

        return audio_data

    def _apply_spectral_gate(
        self, audio_data: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply spectral gating noise reduction using FFT.

        This method uses the magnitude spectrum to gate frequencies below a threshold.
        Optimized to process all channels in parallel.

        Args:
            audio_data: Float audio data, shape (samples, channels)

        Returns:
            Denoised audio data
        """
        # Process all channels at once for better performance
        num_channels = audio_data.shape[1]
        denoised_channels = []

        # For stereo, average channels to build a single noise profile (faster)
        if self.noise_profile is None or self.profile_frames < self.max_profile_frames:
            # Use first channel or average for noise profile
            if num_channels > 1:
                profile_data = audio_data.mean(axis=1)
            else:
                profile_data = audio_data[:, 0]

            profile_spectrum = np.fft.rfft(profile_data)
            profile_magnitude = np.abs(profile_spectrum)

            if self.noise_profile is None:
                self.noise_profile = profile_magnitude
            else:
                # Average with previous profile
                self.noise_profile = (self.noise_profile + profile_magnitude) / 2.0
            self.profile_frames += 1

            if self.verbose and self.profile_frames == self.max_profile_frames:
                self.logger.info("Noise profile calibration complete")

        # Process each channel with the shared noise profile
        for ch in range(num_channels):
            channel_data = audio_data[:, ch]

            # Apply FFT
            spectrum = np.fft.rfft(channel_data)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Apply spectral gate
            if self.noise_profile is not None:
                # Calculate threshold based on noise profile
                threshold = self.noise_profile * (1.0 + self.noise_threshold * 10)

                # Gate: reduce magnitude of frequencies below threshold
                gate_mask = magnitude > threshold
                gated_magnitude = np.where(
                    gate_mask,
                    magnitude,
                    magnitude * self.noise_threshold,  # Reduce but don't completely remove
                )

                # Reconstruct signal
                gated_spectrum = gated_magnitude * np.exp(1j * phase)
                denoised_channel = np.fft.irfft(gated_spectrum, n=len(channel_data))
            else:
                denoised_channel = channel_data

            denoised_channels.append(denoised_channel)

        return np.column_stack(denoised_channels)

    def _apply_wiener_filter(
        self, audio_data: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply Wiener filter for noise reduction.

        This is a simplified Wiener filter implementation using spectral subtraction.

        Args:
            audio_data: Float audio data, shape (samples, channels)

        Returns:
            Denoised audio data
        """
        # Process each channel separately
        denoised_channels = []

        for ch in range(audio_data.shape[1]):
            channel_data = audio_data[:, ch]

            # Apply FFT
            spectrum = np.fft.rfft(channel_data)
            power = np.abs(spectrum) ** 2

            # Build noise power estimate from first few frames
            if self.noise_profile is None or self.profile_frames < self.max_profile_frames:
                if self.noise_profile is None:
                    self.noise_profile = power
                else:
                    self.noise_profile = (self.noise_profile + power) / 2.0
                self.profile_frames += 1

            # Apply Wiener filter
            if self.noise_profile is not None:
                # Wiener gain: signal power / (signal power + noise power)
                # Estimate signal power = total power - noise power
                signal_power = np.maximum(power - self.noise_profile, 0)
                wiener_gain = signal_power / (signal_power + self.noise_profile + 1e-10)

                # Apply gain with smoothing
                wiener_gain = np.clip(wiener_gain, self.noise_threshold, 1.0)

                filtered_spectrum = spectrum * wiener_gain
                denoised_channel = np.fft.irfft(filtered_spectrum, n=len(channel_data))
            else:
                denoised_channel = channel_data

            denoised_channels.append(denoised_channel)

        return np.column_stack(denoised_channels)

    def _apply_normalize(
        self, audio_data: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply volume normalization.

        Args:
            audio_data: Float audio data

        Returns:
            Normalized audio data
        """
        # Calculate current peak level
        peak = np.max(np.abs(audio_data))

        if peak > 1e-6:  # Avoid division by zero
            # Calculate gain to reach target level
            gain = self.target_level / peak

            # Apply gentle limiting to prevent over-amplification
            gain = min(gain, 10.0)  # Max 20dB boost

            return audio_data * gain
        else:
            return audio_data

    def _apply_highpass(
        self, audio_data: npt.NDArray[np.floating[Any]], cutoff: float
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply high-pass filter using FFT.

        Args:
            audio_data: Float audio data, shape (samples, channels)
            cutoff: Cutoff frequency in Hz

        Returns:
            Filtered audio data
        """
        # Process each channel
        filtered_channels = []

        for ch in range(audio_data.shape[1]):
            channel_data = audio_data[:, ch]

            # Apply FFT
            spectrum = np.fft.rfft(channel_data)

            # Calculate frequency bins
            freqs = np.fft.rfftfreq(len(channel_data), 1.0 / self.audio_format.sample_rate)

            # Create high-pass filter (smooth rolloff using sigmoid)
            # This prevents harsh artifacts
            rolloff_width = cutoff * 0.5  # Smooth transition region
            highpass_filter = 1.0 / (1.0 + np.exp(-(freqs - cutoff) / rolloff_width * 10))

            # Apply filter
            filtered_spectrum = spectrum * highpass_filter
            filtered_channel = np.fft.irfft(filtered_spectrum, n=len(channel_data))

            filtered_channels.append(filtered_channel)

        return np.column_stack(filtered_channels)

    def _apply_lowpass(
        self, audio_data: npt.NDArray[np.floating[Any]], cutoff: float
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply low-pass filter using FFT.

        Args:
            audio_data: Float audio data, shape (samples, channels)
            cutoff: Cutoff frequency in Hz

        Returns:
            Filtered audio data
        """
        # Process each channel
        filtered_channels = []

        for ch in range(audio_data.shape[1]):
            channel_data = audio_data[:, ch]

            # Apply FFT
            spectrum = np.fft.rfft(channel_data)

            # Calculate frequency bins
            freqs = np.fft.rfftfreq(len(channel_data), 1.0 / self.audio_format.sample_rate)

            # Create low-pass filter (smooth rolloff using sigmoid)
            rolloff_width = cutoff * 0.5
            lowpass_filter = 1.0 / (1.0 + np.exp((freqs - cutoff) / rolloff_width * 10))

            # Apply filter
            filtered_spectrum = spectrum * lowpass_filter
            filtered_channel = np.fft.irfft(filtered_spectrum, n=len(channel_data))

            filtered_channels.append(filtered_channel)

        return np.column_stack(filtered_channels)

    def process_chunk(self, audio_data: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Process audio chunk with effects.

        Args:
            audio_data: NumPy array of audio samples with shape (samples, channels)

        Returns:
            Processed audio data with effects applied
        """
        original_dtype = audio_data.dtype

        # Convert to float for processing
        processed = self._to_float(audio_data)

        # Apply high-pass filter first (removes rumble)
        if self.highpass is not None:
            processed = self._apply_highpass(processed, self.highpass)

        # Apply low-pass filter (removes hiss)
        if self.lowpass is not None:
            processed = self._apply_lowpass(processed, self.lowpass)

        # Apply noise reduction
        if self.denoise:
            if self.noise_reduction_mode == NoiseReductionMode.SIMPLE:
                processed = self._apply_simple_denoise(processed)
            elif self.noise_reduction_mode == NoiseReductionMode.SPECTRAL_GATE:
                processed = self._apply_spectral_gate(processed)
            elif self.noise_reduction_mode == NoiseReductionMode.WIENER:
                processed = self._apply_wiener_filter(processed)

        # Apply normalization last (after all other effects)
        if self.normalize:
            processed = self._apply_normalize(processed)

        # Convert back to original dtype
        return self._from_float(processed, original_dtype)

    def flush(self) -> npt.NDArray[Any] | None:
        """Flush any remaining data.

        Returns:
            None (no buffering in this pipe)
        """
        return None
