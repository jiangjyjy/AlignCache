from __future__ import annotations

from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _aligncache_sample_video_impl import main


if __name__ == "__main__":
    main(enable_feedback=False)
