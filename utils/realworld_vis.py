"""Diagnostics for real-world EvDehaze outputs (event map + RGB histograms)."""
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_PANEL_RC = {
    "figure.facecolor": "#f7f7f8",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#d8d8dc",
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": "#222222",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
}


def _to_rgb_uint8(img):
    """Accept HxWx3 uint8 RGB or BGR -> RGB uint8."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr


def _event_activity(event):
    if hasattr(event, "detach"):
        ev = event.detach().cpu().numpy()
    else:
        ev = np.asarray(event)
    if ev.ndim == 4:
        ev = ev[0]
    return np.mean(np.abs(ev), axis=0)


def event_tensor_to_vis(event, out_h, out_w):
    """Render event voxel activity as a JET colormap (RGB uint8, HxW)."""
    activity = _event_activity(event)
    assert activity.shape == (out_h, out_w), (
        f"event spatial {activity.shape} != output ({out_h}, {out_w})"
    )
    if activity.max() > activity.min():
        activity = (activity - activity.min()) / (activity.max() - activity.min())
    vis = cv2.applyColorMap((activity * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def event_overlay_on_rgb(hazy_rgb, event, alpha=0.52):
    """Overlay +/− event activity on hazy RGB (easier to judge spatial alignment)."""
    hazy = _to_rgb_uint8(hazy_rgb).astype(np.float32)
    ev = event.detach().cpu().numpy() if hasattr(event, "detach") else np.asarray(event)
    if ev.ndim == 4:
        ev = ev[0]
    h, w = hazy.shape[:2]
    assert ev.shape[1] == h and ev.shape[2] == w
    nb = ev.shape[0] // 2
    ev4 = ev.reshape(nb, 2, h, w)
    pos = ev4[:, 0].mean(0)
    neg = ev4[:, 1].mean(0)
    for m in (pos, neg):
        if m.max() > m.min():
            m[:] = (m - m.min()) / (m.max() - m.min())
    out = hazy.copy()
    out[..., 0] = np.clip(out[..., 0] + pos * 220 * alpha, 0, 255)
    out[..., 2] = np.clip(out[..., 2] + neg * 220 * alpha, 0, 255)
    return out.astype(np.uint8)


def _brightness(img_rgb):
    rgb = _to_rgb_uint8(img_rgb).astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _draw_channel_histograms(axes, hazy_rgb, restored_rgb):
    """Draw 2x2 histograms on a flat iterable of 4 axes."""
    hazy = _to_rgb_uint8(hazy_rgb)
    restored = _to_rgb_uint8(restored_rgb)
    specs = [
        ("Red", "#C0392B", hazy[..., 0].ravel(), restored[..., 0].ravel()),
        ("Green", "#27AE60", hazy[..., 1].ravel(), restored[..., 1].ravel()),
        ("Blue", "#2980B9", hazy[..., 2].ravel(), restored[..., 2].ravel()),
        ("Luma", "#5D6D7E", _brightness(hazy).ravel(), _brightness(restored).ravel()),
    ]
    bins = np.linspace(0, 255, 96)
    for idx, (ax, (name, color, inp, out)) in enumerate(zip(axes, specs)):
        hi, _ = np.histogram(inp, bins=bins)
        ho, _ = np.histogram(out, bins=bins)
        ax.fill_between(bins[:-1], hi, alpha=0.28, color=color, linewidth=0)
        ax.plot(bins[:-1], hi, color=color, linewidth=1.8, linestyle="--", alpha=0.75)
        ax.fill_between(bins[:-1], ho, alpha=0.40, color=color, linewidth=0)
        ax.plot(bins[:-1], ho, color=color, linewidth=2.2, alpha=0.95)
        ax.set_title(name, fontsize=10, fontweight="semibold", pad=6)
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.18, linestyle="-", linewidth=0.6)
        ax.tick_params(labelsize=8)
        if idx >= 2:
            ax.set_xlabel("Pixel value", fontsize=8)
        if idx % 2 == 0:
            ax.set_ylabel("Count", fontsize=8)


def save_channel_histogram(hazy_rgb, restored_rgb, save_path, title=""):
    """Save 2x2 RGB + brightness histograms (input dashed, output solid)."""
    with plt.rc_context(_PANEL_RC):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        if title:
            fig.suptitle(title, fontsize=13, fontweight="semibold", y=0.98)
        _draw_channel_histograms(axes.flat, hazy_rgb, restored_rgb)
        legend = [
            Line2D([0], [0], color="#666", linewidth=1.8, linestyle="--", label="Hazy"),
            Line2D([0], [0], color="#666", linewidth=2.2, label="Dehazed"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def save_triptych(hazy_rgb, event_vis_rgb, restored_rgb, save_path):
    """Hazy | event_vis | restored panel (all same HxW)."""
    hazy = _to_rgb_uint8(hazy_rgb)
    event_vis = _to_rgb_uint8(event_vis_rgb)
    restored = _to_rgb_uint8(restored_rgb)
    h, w = hazy.shape[:2]
    assert event_vis.shape[:2] == (h, w) and restored.shape[:2] == (h, w)
    panel = np.concatenate([hazy, event_vis, restored], axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _imshow_card(ax, img, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=8, color="#222222")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccf")
        spine.set_linewidth(0.8)


def save_combined_panel(
    hazy_rgb, event_vis_rgb, restored_rgb, save_path, title="", event_tensor=None,
):
    """Top: three equal image cards; bottom: full-width 2x2 histogram grid."""
    hazy = _to_rgb_uint8(hazy_rgb)
    restored = _to_rgb_uint8(restored_rgb)
    h, w = hazy.shape[:2]
    if event_tensor is not None:
        middle = event_overlay_on_rgb(hazy, event_tensor)
        mid_title = "Event overlay"
    else:
        middle = _to_rgb_uint8(event_vis_rgb)
        mid_title = "Event"
    assert middle.shape[:2] == (h, w) and restored.shape[:2] == (h, w)

    with plt.rc_context(_PANEL_RC):
        fig = plt.figure(figsize=(13.5, 9.2), facecolor="#f7f7f8")
        if title:
            fig.suptitle(title, fontsize=13, fontweight="semibold", y=0.97, color="#111111")

        gs = fig.add_gridspec(
            2, 1, height_ratios=[1.05, 0.95], hspace=0.18,
            left=0.05, right=0.95, top=0.90, bottom=0.07,
        )
        img_gs = gs[0].subgridspec(1, 3, wspace=0.06)
        for col, (img, label) in enumerate(
            [(hazy, "Hazy input"), (middle, mid_title), (restored, "Dehazed output")]
        ):
            _imshow_card(fig.add_subplot(img_gs[0, col]), img, label)

        hist_gs = gs[1].subgridspec(2, 2, hspace=0.32, wspace=0.22)
        hist_axes = [fig.add_subplot(hist_gs[r, c]) for r in range(2) for c in range(2)]
        _draw_channel_histograms(hist_axes, hazy, restored)

        legend = [
            Line2D([0], [0], color="#666", linewidth=1.8, linestyle="--", label="Hazy"),
            Line2D([0], [0], color="#666", linewidth=2.2, label="Dehazed"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9)
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
