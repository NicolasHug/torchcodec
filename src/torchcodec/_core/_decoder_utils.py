# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
from collections.abc import Sequence
from pathlib import Path

from torch import nn, Tensor
from torchcodec._core._metadata import (
    AudioStreamMetadata,
    get_container_metadata,
    VideoStreamMetadata,
)
from torchcodec._core.ops import (
    add_audio_stream,
    create_from_bytes,
    create_from_file,
    create_from_file_like,
    create_from_tensor,
    set_video_transforms,
)
from torchcodec.transforms import DecoderTransform
from torchcodec.transforms._decoder_transforms import _make_transform_specs


_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


def _create_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    num_ffmpeg_threads: int | None = None,
    dimension_order: str | None = None,
    device: str | None = None,
    device_variant: str | None = None,
    custom_frame_mappings: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    cfm_pts: Tensor | None = None
    cfm_kf: Tensor | None = None
    cfm_dur: Tensor | None = None
    if custom_frame_mappings is not None:
        cfm_pts, cfm_kf, cfm_dur = custom_frame_mappings

    kwargs = dict(
        stream_index=stream_index,
        num_threads=num_ffmpeg_threads,
        dimension_order=dimension_order,
        device=device,
        device_variant=device_variant,
        custom_frame_mappings_pts=cfm_pts,
        custom_frame_mappings_duration=cfm_dur,
        custom_frame_mappings_keyframe_indices=cfm_kf,
    )

    if isinstance(source, str):
        return create_from_file(source, seek_mode, **kwargs)
    elif isinstance(source, Path):
        return create_from_file(str(source), seek_mode, **kwargs)
    elif isinstance(source, io.RawIOBase) or isinstance(source, io.BufferedReader):
        return create_from_file_like(source, seek_mode, **kwargs)
    elif isinstance(source, bytes):
        return create_from_bytes(source, seek_mode, **kwargs)
    elif isinstance(source, Tensor):
        return create_from_tensor(source, seek_mode, **kwargs)
    elif isinstance(source, io.TextIOBase):
        raise TypeError(
            "source is for reading text, likely from open(..., 'r'). Try with 'rb' for binary reading?"
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        return create_from_file_like(source, seek_mode, **kwargs)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes, Tensor and file-like objects with "
        "read(self, size: int) -> bytes and "
        "seek(self, offset: int, whence: int) -> int methods."
    )


def create_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
) -> Tensor:
    return _create_decoder(source=source, seek_mode=seek_mode)


def create_audio_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> tuple[Tensor, int, AudioStreamMetadata]:

    decoder = create_decoder(source=source, seek_mode=seek_mode)

    container_metadata = get_container_metadata(decoder)

    if stream_index is None:
        stream_index = container_metadata.best_audio_stream_index
        if stream_index is None:
            raise ValueError(
                "The best audio stream is unknown and there is no specified stream. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )

    if stream_index >= len(container_metadata.streams):
        raise ValueError(f"The stream at index {stream_index} is not a valid stream.")

    metadata = container_metadata.streams[stream_index]
    if not isinstance(metadata, AudioStreamMetadata):
        raise ValueError(f"The stream at index {stream_index} is not an audio stream.")

    add_audio_stream(
        decoder,
        stream_index=stream_index,
        sample_rate=sample_rate,
        num_channels=num_channels,
    )

    return (decoder, stream_index, metadata)


def _get_and_validate_stream_metadata(
    *,
    decoder: Tensor,
    stream_index: int | None = None,
) -> tuple[VideoStreamMetadata, int]:
    container_metadata = get_container_metadata(decoder)

    if stream_index is None:
        if (stream_index := container_metadata.best_video_stream_index) is None:
            raise ValueError(
                "The best video stream is unknown and there is no specified stream. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )

    if stream_index >= len(container_metadata.streams):
        raise ValueError(f"The stream index {stream_index} is not a valid stream.")

    metadata = container_metadata.streams[stream_index]
    if not isinstance(metadata, VideoStreamMetadata):
        raise ValueError(f"The stream at index {stream_index} is not a video stream. ")

    if metadata.begin_stream_seconds is None:
        raise ValueError(
            "The minimum pts value in seconds is unknown. "
            + _ERROR_REPORTING_INSTRUCTIONS
        )

    if metadata.end_stream_seconds is None:
        raise ValueError(
            "The maximum pts value in seconds is unknown. "
            + _ERROR_REPORTING_INSTRUCTIONS
        )

    if metadata.num_frames is None:
        raise ValueError(
            "The number of frames is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
        )

    return (
        metadata,
        stream_index,
    )


def create_video_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    dimension_order: str = "NCHW",
    num_ffmpeg_threads: int = 1,
    device: str,
    device_variant: str = "ffmpeg",
    transforms: Sequence[DecoderTransform | nn.Module] | None = None,
    custom_frame_mappings: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, VideoStreamMetadata, int]:
    """Create a video decoder and add a video stream.

    This function creates a decoder, adds a video stream, and optionally sets
    transforms. The video stream is added during construction. Transforms are
    set separately afterwards since they may depend on the stream's H/W
    dimensions (e.g. RandomCrop).

    Args:
        source: The source of the video.
        seek_mode: The seek mode for the decoder.
        stream_index: The stream index to decode, or None to use the best stream.
        dimension_order: The dimension order for decoded frames.
        num_ffmpeg_threads: Number of FFmpeg threads for CPU decoding.
        device: The device for decoding (as a string).
        device_variant: The CUDA backend variant to use ("ffmpeg" or "beta").
        transforms: Optional sequence of transforms to apply.
        custom_frame_mappings: Optional pre-processed frame mappings data.

    Returns:
        A tuple of (decoder, metadata, stream_index).
    """
    # Step 1: Create a lightweight decoder (no video stream) to validate
    # stream_index and resolve it if None.
    validation_decoder = _create_decoder(
        source=source,
        seek_mode=seek_mode,
    )

    (
        metadata,
        stream_index,
    ) = _get_and_validate_stream_metadata(
        decoder=validation_decoder, stream_index=stream_index
    )

    # Delete the validation decoder before creating the actual one, so that
    # file-like sources can be re-read from the beginning.
    del validation_decoder
    if hasattr(source, "seek"):
        source.seek(0, 0)

    # Step 2: Create the actual decoder with video stream
    decoder = _create_decoder(
        source=source,
        seek_mode=seek_mode,
        stream_index=stream_index,
        num_ffmpeg_threads=num_ffmpeg_threads,
        dimension_order=dimension_order,
        device=device,
        device_variant=device_variant,
        custom_frame_mappings=custom_frame_mappings,
    )

    # Re-query metadata from the actual decoder (which has the stream added
    # and may have scanned the file for exact seek mode).
    (
        metadata,
        stream_index,
    ) = _get_and_validate_stream_metadata(
        decoder=decoder, stream_index=stream_index
    )

    # Step 3: Set transforms using H/W (Python RNG preserved)
    transform_specs = _make_transform_specs(
        transforms,
        input_dims=(metadata.height, metadata.width),
    )
    if transform_specs:
        set_video_transforms(decoder, transform_specs)

    return (
        decoder,
        metadata,
        stream_index,
    )
