"""
=============================================================
  High-Quality Batch Image Compressor
  Uses: pyvips (libvips) + MozJPEG
  Features:
    - Max file size (KB/MB) enforcement via binary search
    - Format conversion (JPEG, WebP, AVIF, PNG)
    - Multithreaded processing (~5-10x faster)
    - Progress bar
    - Detailed log file
    - Skips already-small images
    - Preserves folder structure (optional)
=============================================================
"""

import os
import io
import time
import logging
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Optional: rich progress bar (falls back to plain print) ──────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Try pyvips first, fall back to Pillow ────────────────────────────────────
try:
    import pyvips
    HAS_PYVIPS = True
except ImportError:
    HAS_PYVIPS = False

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

if not HAS_PYVIPS and not HAS_PILLOW:
    raise ImportError(
        "No image library found.\n"
        "Install at least one:\n"
        "  pip install pyvips\n"
        "  pip install Pillow"
    )

# ─────────────────────────────────────────────────────────────────────────────
#   ⚙️  DEFAULT CONFIG  — override via CLI args or edit here
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT = dict(
    input_folder   = "input_images",   # Folder with your original images
    output_folder  = "output_images",  # Where compressed images go
    output_format  = "webp",           # webp | jpeg | avif | png
    quality        = 82,               # Starting quality (1-100)
    max_size_kb    = None,             # e.g. 300  → enforce ≤ 300 KB; None = no limit
    min_quality    = 20,               # Never compress below this quality
    threads        = 4,                # Parallel workers (increase for more CPU cores)
    preserve_tree  = False,            # True = keep sub-folder structure in output
    skip_small     = True,             # Skip images already within max_size_kb
    strip_metadata = True,             # Remove EXIF/GPS/camera data (saves ~5-15 KB)
    log_file       = "compress_log.txt",
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}

# ─────────────────────────────────────────────────────────────────────────────
#   LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("compressor")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    # File handler — full detail
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
#   PYVIPS BACKEND  (preferred — faster, MozJPEG quality)
# ─────────────────────────────────────────────────────────────────────────────
def _vips_to_bytes(img: "pyvips.Image", fmt: str, quality: int, strip: bool) -> bytes:
    """Encode a pyvips image to bytes at the given quality."""
    kwargs = dict(Q=quality, strip=strip)
    if fmt == "webp":
        kwargs["lossless"] = False
        kwargs["effort"] = 0  # 0=fastest (equivalent to method=0 in Pillow)
    elif fmt == "avif":
        kwargs["compression"] = "av1"
    buf = img.write_to_buffer(f".{fmt}", **kwargs)
    return buf


def compress_pyvips(input_path: Path, output_path: Path, cfg: dict, logger) -> dict:
    """
    Compress a single image using pyvips.
    Returns a result dict with original_kb, final_kb, quality_used, status.
    """
    original_kb = input_path.stat().st_size / 1024
    fmt         = cfg["output_format"]
    strip       = cfg["strip_metadata"]
    max_kb      = cfg["max_size_kb"]
    min_q       = cfg["min_quality"]
    start_q     = cfg["quality"]
    min_kb      = cfg.get("min_size_kb", 0) # Fallback to 0 if not set

    try:
        # sequential access = lower RAM usage for large images
        img = pyvips.Image.new_from_file(str(input_path), access="sequential")

        # ── No size limit or Lossless Format: just compress at fixed quality ──
        if fmt == "png" or not max_kb:
            buf = _vips_to_bytes(img, fmt, start_q, strip)
            output_path.write_bytes(buf)
            return dict(status="ok", original_kb=original_kb,
                        final_kb=len(buf)/1024, quality_used=start_q)

        # ── Already small enough: skip or just convert format ─────────────────
        if cfg["skip_small"] and original_kb <= max_kb and original_kb >= min_kb:
            buf = _vips_to_bytes(img, fmt, start_q, strip)
            output_path.write_bytes(buf)
            return dict(status="skipped_small", original_kb=original_kb,
                        final_kb=len(buf)/1024, quality_used=start_q)

        # ── Fast path: try start_q first ──────────────────────────────────────
        if max_kb and fmt == "webp":
            buf = _vips_to_bytes(img, fmt, start_q, strip)
            size_kb = len(buf) / 1024
            if min_kb <= size_kb <= max_kb:
                output_path.write_bytes(buf)
                return dict(status="ok", original_kb=original_kb,
                            final_kb=size_kb, quality_used=start_q)
            
            # Narrow the search range based on pre-check
            if size_kb > max_kb:
                lo, hi = min_q, start_q - 1
            else:
                lo, hi = start_q + 1, 95
        else:
            lo, hi = min_q, 95

        # ── Binary search for target size ─────────────────────────────────────
        best_buf   = None
        best_q     = min_q

        while lo <= hi:
            mid = (lo + hi) // 2
            # Re-open for each pass (sequential access can only be read once)
            img = pyvips.Image.new_from_file(str(input_path), access="sequential")
            buf = _vips_to_bytes(img, fmt, mid, strip)
            size_kb = len(buf) / 1024

            if size_kb > max_kb:
                hi = mid - 1         # too big, lower quality
            else:
                best_buf = buf
                best_q   = mid
                lo       = mid + 1   # try higher quality (might still fit)
                if size_kb >= min_kb and size_kb <= max_kb:
                    pass # We are in the sweet spot! but we'll try to push quality higher as long as size_kb <= max_kb

        if best_buf:
            output_path.write_bytes(best_buf)
            return dict(status="ok", original_kb=original_kb,
                        final_kb=len(best_buf)/1024, quality_used=best_q)
        else:
            # Even min quality didn't fit — save at min quality anyway
            img = pyvips.Image.new_from_file(str(input_path), access="sequential")
            buf = _vips_to_bytes(img, fmt, min_q, strip)
            output_path.write_bytes(buf)
            return dict(status="warn_over_limit", original_kb=original_kb,
                        final_kb=len(buf)/1024, quality_used=min_q)

    except Exception as e:
        logger.error(f"pyvips error on {input_path.name}: {e}")
        return dict(status="error", original_kb=original_kb,
                    final_kb=0, quality_used=0, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#   PILLOW BACKEND  (fallback)
# ─────────────────────────────────────────────────────────────────────────────
_PIL_FORMAT = {"webp": "WEBP", "jpeg": "JPEG", "jpg": "JPEG",
               "png": "PNG", "avif": "AVIF"}

def compress_pillow(input_path: Path, output_path: Path, cfg: dict, logger) -> dict:
    original_kb = input_path.stat().st_size / 1024
    pil_fmt     = _PIL_FORMAT.get(cfg["output_format"], "JPEG")
    max_kb      = cfg["max_size_kb"]
    min_q       = cfg["min_quality"]
    start_q     = cfg["quality"]
    min_kb      = cfg.get("min_size_kb", 0)

    def _encode(img, quality) -> bytes:
        buf = io.BytesIO()
        save_kwargs = dict(format=pil_fmt, quality=quality, optimize=True)
        if pil_fmt == "JPEG":
            save_kwargs["progressive"] = True
        
        if pil_fmt == "WEBP":
            save_kwargs["method"] = 0  # 0=fastest, matches benchmark's speedup
        
        # Pillow's PNG doesn't use 'quality'. It only uses 'compress_level' (0-9).
        if pil_fmt == "PNG":
            save_kwargs.pop("quality", None)
            
        img.save(buf, **save_kwargs)
        return buf.getvalue()

    try:
        img = Image.open(input_path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if pil_fmt == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")

        # ── Binary search doesn't work for lossless PNGs ──────────────────────
        if pil_fmt == "PNG" or not max_kb:
            raw = _encode(img, start_q)
            output_path.write_bytes(raw)
            return dict(status="ok", original_kb=original_kb,
                        final_kb=len(raw)/1024, quality_used=start_q)

        if cfg["skip_small"] and original_kb <= max_kb and original_kb >= min_kb:
            raw = _encode(img, start_q)
            output_path.write_bytes(raw)
            return dict(status="skipped_small", original_kb=original_kb,
                        final_kb=len(raw)/1024, quality_used=start_q)

        # ── Fast path: try start_q first ──────────────────────────────────────
        if max_kb and pil_fmt == "WEBP":
            raw = _encode(img, start_q)
            size_kb = len(raw) / 1024
            if min_kb <= size_kb <= max_kb:
                output_path.write_bytes(raw)
                return dict(status="ok", original_kb=original_kb,
                            final_kb=size_kb, quality_used=start_q)
            
            # Narrow the search range based on pre-check
            if size_kb > max_kb:
                lo, hi = min_q, start_q - 1
            else:
                lo, hi = start_q + 1, 95
        else:
            lo, hi = min_q, 95

        best_raw = None
        best_q   = min_q

        while lo <= hi:
            mid     = (lo + hi) // 2
            raw     = _encode(img, mid)
            size_kb = len(raw) / 1024

            if size_kb > max_kb:
                hi = mid - 1
            else:
                best_raw = raw
                best_q   = mid
                lo       = mid + 1

        if best_raw:
            output_path.write_bytes(best_raw)
            return dict(status="ok", original_kb=original_kb,
                        final_kb=len(best_raw)/1024, quality_used=best_q)
        else:
            # ── Phase 3: Resize Fallback ──────────────────────────────────────
            scales = [0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
            for scale in scales:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Check with starting quality on this resize
                raw = _encode(resized_img, start_q)
                size_kb = len(raw) / 1024
                
                if min_kb <= size_kb <= max_kb:
                    output_path.write_bytes(raw)
                    return dict(status="ok (resized)", original_kb=original_kb,
                                final_kb=size_kb, quality_used=start_q)
                                
                if size_kb < min_kb:
                    continue  # We shunk it so much it's too small, go to next scale or fallback
                    
                # Binary search quality on the resized image
                r_lo, r_hi = min_q, start_q - 1
                r_best = None
                r_best_q = min_q
                
                while r_lo <= r_hi:
                    r_mid = (r_lo + r_hi) // 2
                    r_raw = _encode(resized_img, r_mid)
                    r_size_kb = len(r_raw) / 1024
                    
                    if r_size_kb > max_kb:
                        r_hi = r_mid - 1
                    else:
                        r_best = r_raw
                        r_best_q = r_mid
                        r_lo = r_mid + 1
                            
                if r_best:
                    output_path.write_bytes(r_best)
                    return dict(status="ok (resized)", original_kb=original_kb,
                                final_kb=len(r_best)/1024, quality_used=r_best_q)

            # ── Absolute fallback (if even resize fails) ──────────────────────
            raw = _encode(img, min_q)
            output_path.write_bytes(raw)
            return dict(status="warn_over_limit", original_kb=original_kb,
                        final_kb=len(raw)/1024, quality_used=min_q)

    except Exception as e:
        logger.error(f"Pillow error on {input_path.name}: {e}")
        return dict(status="error", original_kb=original_kb,
                    final_kb=0, quality_used=0, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#   WORKER — called per image inside thread pool
# ─────────────────────────────────────────────────────────────────────────────
def _process_one(args):
    input_path, output_path, cfg, logger = args
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PYVIPS:
        return input_path, compress_pyvips(input_path, output_path, cfg, logger)
    else:
        return input_path, compress_pillow(input_path, output_path, cfg, logger)


# ─────────────────────────────────────────────────────────────────────────────
#   MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(cfg: dict):
    in_dir  = Path(cfg["input_folder"])
    out_dir = Path(cfg["output_folder"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(cfg["log_file"])
    logger.info("=" * 60)
    logger.info("Starting compression run")
    logger.info(f"Config: {cfg}")

    # ── Collect all image files ───────────────────────────────────────────────
    all_files = [
        p for p in in_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    total = len(all_files)
    if total == 0:
        print(f"❌  No supported images found in '{in_dir}'")
        return

    ext = cfg["output_format"]
    print(f"\n{'─'*55}")
    print(f"  Found        : {total} images")
    print(f"  Output format: {ext.upper()}")
    print(f"  Quality      : {cfg['quality']}")
    print(f"  Max size     : {cfg['max_size_kb']} KB" if cfg["max_size_kb"] else "  Max size     : (no limit)")
    print(f"  Threads      : {cfg['threads']}")
    print(f"  Backend      : {'pyvips (libvips)' if HAS_PYVIPS else 'Pillow'}")
    print(f"{'─'*55}\n")

    # ── Build work list ───────────────────────────────────────────────────────
    work = []
    for p in all_files:
        if cfg["preserve_tree"]:
            rel      = p.relative_to(in_dir)
            out_path = out_dir / rel.with_suffix(f".{ext}")
        else:
            out_path = out_dir / (p.stem + f".{ext}")
        work.append((p, out_path, cfg, logger))

    # ── Stats counters ────────────────────────────────────────────────────────
    results = {"ok": 0, "skipped_small": 0, "warn_over_limit": 0, "error": 0}
    total_saved_kb = 0.0
    file_details = []
    lock = threading.Lock()

    # ── Thread pool ───────────────────────────────────────────────────────────
    iterator = ThreadPoolExecutor(max_workers=cfg["threads"]).map(_process_one, work)

    if HAS_TQDM:
        iterator = tqdm(iterator, total=total, unit="img", ncols=72)

    t0 = time.time()
    for input_path, result in iterator:
        status     = result["status"]
        orig_kb    = result["original_kb"]
        final_kb   = result["final_kb"]
        quality    = result["quality_used"]
        saved      = orig_kb - final_kb

        with lock:
            results[status] = results.get(status, 0) + 1
            if status != "error":
                total_saved_kb += saved
            
            file_details.append({
                "File Name": input_path.name,
                "Original Format": input_path.suffix.upper().replace(".", ""),
                "New Format": ext.upper(),
                "Original Size (KB)": round(orig_kb, 1),
                "New Size (KB)": round(final_kb, 1),
                "Status": status
            })

        # Console line (suppressed when tqdm is active)
        icon = {"ok": "✅", "skipped_small": "⏭ ", "warn_over_limit": "⚠️ ", "error": "❌"}.get(status, "  ")
        msg  = f"{icon} {input_path.name:<40} {orig_kb:>8.1f} KB → {final_kb:>8.1f} KB  (q={quality})"
        if not HAS_TQDM:
            print(msg)
        logger.info(msg)

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  ✅  Compressed     : {results.get('ok', 0)}")
    print(f"  ⏭   Already small  : {results.get('skipped_small', 0)}")
    print(f"  ⚠️   Over limit     : {results.get('warn_over_limit', 0)}")
    print(f"  ❌  Errors         : {results.get('error', 0)}")
    print(f"  💾  Total saved    : {total_saved_kb/1024:.1f} MB  ({total_saved_kb:.0f} KB)")
    print(f"  ⏱   Time           : {elapsed:.1f}s  ({total/elapsed:.1f} img/s)")
    print(f"  📁  Output folder  : {out_dir.resolve()}")
    print(f"  📄  Log file       : {Path(cfg['log_file']).resolve()}")
    print(f"{'═'*55}\n")

    logger.info(f"Done. saved={total_saved_kb:.0f} KB  time={elapsed:.1f}s")

    return file_details


# ─────────────────────────────────────────────────────────────────────────────
#   CLI  (optional — you can also just edit DEFAULT above and run directly)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="High-quality batch image compressor (pyvips / Pillow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input",   default=DEFAULT["input_folder"],  help="Input folder")
    p.add_argument("-o", "--output",  default=DEFAULT["output_folder"], help="Output folder")
    p.add_argument("-f", "--format",  default=DEFAULT["output_format"],
                   choices=["webp", "jpeg", "avif", "png"],             help="Output format")
    p.add_argument("-q", "--quality", default=DEFAULT["quality"],  type=int, help="Quality (1-100)")
    p.add_argument("--max-kb",        default=DEFAULT["max_size_kb"], type=float,
                   help="Maximum output file size in KB (e.g. 300). Omit for no limit.")
    p.add_argument("--min-quality",   default=DEFAULT["min_quality"],  type=int,
                   help="Never go below this quality when enforcing max-kb")
    p.add_argument("--threads",       default=DEFAULT["threads"],       type=int,
                   help="Number of parallel worker threads")
    p.add_argument("--preserve-tree", action="store_true",
                   default=DEFAULT["preserve_tree"],  help="Keep sub-folder structure")
    p.add_argument("--no-strip",      action="store_true",
                   help="Keep EXIF/metadata (by default it is stripped)")
    p.add_argument("--log",           default=DEFAULT["log_file"],      help="Log file path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = dict(
        input_folder   = args.input,
        output_folder  = args.output,
        output_format  = args.format,
        quality        = args.quality,
        max_size_kb    = args.max_kb,
        min_quality    = args.min_quality,
        threads        = args.threads,
        preserve_tree  = args.preserve_tree,
        skip_small     = DEFAULT["skip_small"],
        strip_metadata = not args.no_strip,
        log_file       = args.log,
    )
    run(cfg)
