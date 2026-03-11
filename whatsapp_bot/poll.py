"""
Blocking file watcher for the WhatsApp incoming queue.
Uses inotify (Linux) to block until a new .json file appears.
Prints the filename and exits — designed to be called in a loop by Claude Code.
"""

import sys
import time
from pathlib import Path

QUEUE_DIR = Path(__file__).parent / "queue" / "incoming"


def poll_inotify(timeout: int = 60) -> str | None:
    """Use inotify to wait for a new file. Returns filename or None on timeout."""
    try:
        import inotify.adapters
    except ImportError:
        return None

    # Check if files already exist before waiting
    existing = list(QUEUE_DIR.glob("*.json"))
    if existing:
        return existing[0].name

    i = inotify.adapters.Inotify()
    i.add_watch(str(QUEUE_DIR), mask=0x00000100 | 0x00000080)  # IN_CREATE | IN_MOVED_TO

    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(1, int(deadline - time.time()))
        for event in i.event_gen(timeout_s=remaining, yield_nones=False):
            if event is not None:
                _, type_names, path, filename = event
                if filename and filename.endswith(".json"):
                    return filename
        # Also check glob in case we missed the event
        existing = list(QUEUE_DIR.glob("*.json"))
        if existing:
            return existing[0].name

    return None


def poll_fallback(timeout: int = 60) -> str | None:
    """Fallback: poll with long sleeps."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = list(QUEUE_DIR.glob("*.json"))
        if files:
            return files[0].name
        time.sleep(2)
    return None


def main():
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    # Try inotify first, fall back to polling
    try:
        import inotify.adapters
        result = poll_inotify(timeout)
    except ImportError:
        result = poll_fallback(timeout)

    if result:
        print(result)
    else:
        print("TIMEOUT")


if __name__ == "__main__":
    main()
