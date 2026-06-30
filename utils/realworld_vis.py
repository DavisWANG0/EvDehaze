"""Diagnostics for real-world EvDehaze outputs (event map + RGB histograms)."""
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _to_rgb_uint8(img):
    """Accept HxWx3 uint8 RGB or BGR -> RGB uint8."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr


def event_tensor_to_vis(event, out_h, out_w):
    """Render event voxel activity as a JET colormap (RGB uint8)."""
    if hasattr(event, "detach"):
        ev = event.detach().cpu().numpy()
    else:
        ev = np.asarray(event)
    if ev.ndim == 4:
        ev = ev[0]
    activity = np.mean(np.abs(ev), axis=0)
    if activity.max() > activity.min():
        activity = (activity - activity.min()) / (activity.max() - activity.min())
    vis = cv2.applyColorMap((activity * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    if vis.shape[0] != out_h or vis.shape[1] != out_w:
        vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return vis


def _brightness(img_rgb):
    rgb = _to_rgb_uint8(img_rgb).astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def save_channel_histogram(hazy_rgb, restored_rgb, save_path, title=""):
    """Save 2x2 RGB + brightness histograms (input dashed, output solid)."""
    hazy = _to_rgb_uint8(hazy_rgb)
    restored = _to_rgb_uint8(restored_rgb)
    if hazy.shape != restored.shape:
        restored = cv2.resize(restored, (hazy.shape[1], hazy.shape[0]), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    specs = [
        (0, "Red", "#E74C3C", hazy[..., 0].ravel(), restored[..., 0].ravel()),
        (1, "Green", "#2ECC71", hazy[..., 1].ravel(), restored[..., 1].ravel()),
        (2, "Blue", "#3498DB", hazy[..., 2].ravel(), restored[..., 2].ravel()),
        (3, "Brightness", "#757575", _brightness(hazy).ravel(), _brightness(restored).ravel()),
    ]
    bins = np.linspace(0, 255, 128)
    for ax_idx, name, color, inp, out in specs:
        ax = axes.flat[ax_idx]
        hi, _ = np.histogram(inp, bins=bins)
        ho, _ = np.histogram(out, bins=bins)
        ax.fill_between(bins[:-1], hi, alpha=0.35, color=color)
        ax.plot(bins[:-1], hi, color=color, linewidth=2, linestyle="--", alpha=0.85)
        ax.fill_between(bins[:-1], ho, alpha=0.55, color=color)
        ax.plot(bins[:-1], ho, color=color, linewidth=2.5, alpha=1.0)
        ax.set_title(f"{name} Channel", fontsize=12, fontweight="bold")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, linestyle="--")

    legend = [
        Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="INPUT (Hazy)"),
        Line2D([0], [0], color="gray", linewidth=2.5, label="OUTPUT (Dehazed)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=True, fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_triptych(hazy_rgb, event_vis_rgb, restored_rgb, save_path):
    """Hazy | event_vis | restored panel for video frames."""
    hazy = _to_rgb_uint8(hazy_rgb)
    event_vis = _to_rgb_uint8(event_vis_rgb)
    restored = _to_rgb_uint8(restored_rgb)
    h, w = hazy.shape[:2]
    if event_vis.shape[:2] != (h, w):
        event_vis = cv2.resize(event_vis, (w, h), interpolation=cv2.INTER_LINEAR)
    if restored.shape[:2] != (h, w):
        restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_LINEAR)
    panel = np.concatenate([hazy, event_vis, restored], axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
