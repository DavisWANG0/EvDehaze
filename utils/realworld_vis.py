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
    """Render event voxel activity as a JET colormap (RGB uint8, HxW)."""
    if hasattr(event, "detach"):
        ev = event.detach().cpu().numpy()
    else:
        ev = np.asarray(event)
    if ev.ndim == 4:
        ev = ev[0]
    assert ev.shape[1] == out_h and ev.shape[2] == out_w, (
        f"event spatial {ev.shape[1:]} != output ({out_h}, {out_w})"
    )
    activity = np.mean(np.abs(ev), axis=0)
    if activity.max() > activity.min():
        activity = (activity - activity.min()) / (activity.max() - activity.min())
    vis = cv2.applyColorMap((activity * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis


def _brightness(img_rgb):
    rgb = _to_rgb_uint8(img_rgb).astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _draw_channel_histograms(axes, hazy_rgb, restored_rgb):
    """Draw 2x2 histograms on a flat iterable of 4 axes."""
    hazy = _to_rgb_uint8(hazy_rgb)
    restored = _to_rgb_uint8(restored_rgb)
    specs = [
        ("Red", "#E74C3C", hazy[..., 0].ravel(), restored[..., 0].ravel()),
        ("Green", "#2ECC71", hazy[..., 1].ravel(), restored[..., 1].ravel()),
        ("Blue", "#3498DB", hazy[..., 2].ravel(), restored[..., 2].ravel()),
        ("Brightness", "#757575", _brightness(hazy).ravel(), _brightness(restored).ravel()),
    ]
    bins = np.linspace(0, 255, 128)
    for ax, (name, color, inp, out) in zip(axes, specs):
        hi, _ = np.histogram(inp, bins=bins)
        ho, _ = np.histogram(out, bins=bins)
        ax.fill_between(bins[:-1], hi, alpha=0.35, color=color)
        ax.plot(bins[:-1], hi, color=color, linewidth=2, linestyle="--", alpha=0.85)
        ax.fill_between(bins[:-1], ho, alpha=0.55, color=color)
        ax.plot(bins[:-1], ho, color=color, linewidth=2.5, alpha=1.0)
        ax.set_title(f"{name} Channel", fontsize=11, fontweight="bold")
        ax.set_xlabel("Pixel Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, linestyle="--")


def save_channel_histogram(hazy_rgb, restored_rgb, save_path, title=""):
    """Save 2x2 RGB + brightness histograms (input dashed, output solid)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    _draw_channel_histograms(axes.flat, hazy_rgb, restored_rgb)
    legend = [
        Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="INPUT (Hazy)"),
        Line2D([0], [0], color="gray", linewidth=2.5, label="OUTPUT (Dehazed)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=True, fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_triptych(hazy_rgb, event_vis_rgb, restored_rgb, save_path):
    """Hazy | event_vis | restored panel (all same HxW)."""
    hazy = _to_rgb_uint8(hazy_rgb)
    event_vis = _to_rgb_uint8(event_vis_rgb)
    restored = _to_rgb_uint8(restored_rgb)
    h, w = hazy.shape[:2]
    assert event_vis.shape[:2] == (h, w) and restored.shape[:2] == (h, w), (
        f"triptych size mismatch: hazy {hazy.shape[:2]}, "
        f"event {event_vis.shape[:2]}, restored {restored.shape[:2]}"
    )
    panel = np.concatenate([hazy, event_vis, restored], axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def save_combined_panel(hazy_rgb, event_vis_rgb, restored_rgb, save_path, title=""):
    """Triptych on top + 2x2 channel histograms below, single figure."""
    hazy = _to_rgb_uint8(hazy_rgb)
    event_vis = _to_rgb_uint8(event_vis_rgb)
    restored = _to_rgb_uint8(restored_rgb)
    h, w = hazy.shape[:2]
    assert event_vis.shape[:2] == (h, w) and restored.shape[:2] == (h, w)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0], hspace=0.35, wspace=0.25)

    for col, (img, label) in enumerate(
        [(hazy, "INPUT (Hazy)"), (event_vis, "Event"), (restored, "OUTPUT (Dehazed)")]
    ):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.add_subplot(gs[0, 3]).axis("off")

    hist_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    hist_axes += [fig.add_subplot(gs[2, i]) for i in range(2)]
    _draw_channel_histograms(hist_axes, hazy, restored)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    legend = [
        Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="INPUT (Hazy)"),
        Line2D([0], [0], color="gray", linewidth=2.5, label="OUTPUT (Dehazed)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=True, fontsize=10)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
