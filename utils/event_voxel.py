"""Convert raw event streams (.npy with x,y,p,t) to voxel grids for EvDehaze."""
from pathlib import Path

import numpy as np

# SOTS synthetic events span ~83 ms per clear frame (t in nanoseconds).
SOTS_EVENT_WINDOW_MS = 83
# Real-world event npy files cover ~1 s (t in microseconds).
REALWORLD_EVENT_SPAN_MS = 1000


def parse_frame_ms(rgb_path):
    """Parse frame offset in ms from ``<prefix>_<session>_<frame_ms>.png``."""
    return int(Path(rgb_path).stem.split("_")[-1])


def parse_event_chunk_ms(event_path):
    """Parse sync offset from ``<prefix>_<session>_<chunk>_event.npy`` (e.g. 7 ms)."""
    return int(Path(event_path).stem.split("_")[2])


def frame_event_offset_ms(frame_ms, event_chunk_ms, use_modulo=False):
    """Map RGB ``frame_ms`` to event-stream offset (ms from stream start).

    Session naming: RGB burst starts at session start; event recorder starts
    ``event_chunk_ms`` later (filename suffix, typically ``0007`` → 7 ms).
    """
    if use_modulo:
        return (frame_ms % REALWORLD_EVENT_SPAN_MS) - event_chunk_ms
    return frame_ms - event_chunk_ms


def is_frame_alignable(
    frame_ms,
    event_chunk_ms,
    window_ms=SOTS_EVENT_WINDOW_MS,
    event_span_ms=REALWORLD_EVENT_SPAN_MS,
    use_modulo=False,
):
    """True if the full SOTS-length window lies inside the event clip."""
    off = frame_event_offset_ms(frame_ms, event_chunk_ms, use_modulo=use_modulo)
    if use_modulo:
        off = off % event_span_ms
    return off >= 0 and off + window_ms <= event_span_ms


def crop_events_temporal(
    events,
    frame_ms,
    event_chunk_ms=0,
    window_ms=SOTS_EVENT_WINDOW_MS,
    use_modulo=False,
    event_span_ms=REALWORLD_EVENT_SPAN_MS,
):
    """Keep events for one RGB exposure: aligned ~83 ms window like SOTS.

    Alignment: RGB ``frame_ms`` is ms from session start; event stream ``t0``
    matches session start; recorder begins ``event_chunk_ms`` later.
    Window: ``[frame_ms - chunk, frame_ms - chunk + 83)`` in event time.
    """
    t = events["t"].astype(np.float64)
    if len(t) == 0:
        return events
    t0 = t.min()
    off_ms = frame_event_offset_ms(frame_ms, event_chunk_ms, use_modulo=use_modulo)
    if use_modulo:
        off_ms = off_ms % event_span_ms
    off_ms = float(np.clip(off_ms, 0, max(event_span_ms - 1, 0)))
    win_ms = min(max(int(window_ms), 1), int(event_span_ms - off_ms))
    lo = t0 + off_ms * 1000.0
    hi = t0 + (off_ms + win_ms) * 1000.0
    mask = (t >= lo) & (t < hi)
    return events[mask]


def events_to_voxel_grid(events, num_bins=4, height=None, width=None, normalize=True):
    """Rasterize structured event array to (num_bins, 2, H, W)."""
    t = events['t'].astype(np.float64)
    x = events['x'].astype(np.int64)
    y = events['y'].astype(np.int64)
    p = events['p'].astype(np.int64)

    if height is None:
        height = int(y.max()) + 1
    if width is None:
        width = int(x.max()) + 1

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]
    if len(t) == 0:
        return np.zeros((num_bins, 2, height, width), dtype=np.float32)

    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_idx = ((t - t_min) / (t_max - t_min) * (num_bins - 1e-6)).astype(np.int64)
    else:
        t_idx = np.zeros(len(t), dtype=np.int64)
    t_idx = np.clip(t_idx, 0, num_bins - 1)

    pol = (p > 0).astype(np.int64)
    voxel = np.zeros((num_bins, 2, height, width), dtype=np.float32)
    np.add.at(voxel, (t_idx, pol, y, x), 1.0)

    if normalize and voxel.max() > 0:
        voxel /= voxel.max() + 1e-6
    return voxel


def load_event_voxel_for_rgb(
    event_path,
    rgb_h,
    rgb_w,
    num_bins=4,
    frame_ms=None,
    event_chunk_ms=0,
    window_ms=SOTS_EVENT_WINDOW_MS,
    time_crop=False,
    use_modulo=False,
):
    """Load raw .npy events and return (8, rgb_h, rgb_w) tensor-ready numpy."""
    events = np.load(event_path)
    if time_crop and frame_ms is not None:
        events = crop_events_temporal(
            events,
            frame_ms,
            event_chunk_ms=event_chunk_ms,
            window_ms=window_ms,
            use_modulo=use_modulo,
        )
    if len(events) == 0:
        return np.zeros((num_bins * 2, rgb_h, rgb_w), dtype=np.float32)
    eh = int(events['y'].max()) + 1
    ew = int(events['x'].max()) + 1
    voxel = events_to_voxel_grid(events, num_bins=num_bins, height=eh, width=ew)
    if (eh, ew) != (rgb_h, rgb_w):
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(voxel).unsqueeze(0)
        t = t.reshape(1, num_bins * 2, eh, ew)
        t = F.interpolate(t, size=(rgb_h, rgb_w), mode='bilinear', align_corners=False)
        voxel = t.squeeze(0).reshape(num_bins, 2, rgb_h, rgb_w).numpy()
    return voxel.reshape(num_bins * 2, rgb_h, rgb_w).astype(np.float32)
