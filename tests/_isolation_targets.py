"""Child-process entry points for tests/test_isolated.py.

They live in an importable module (not inside a test function) because the child is
started with the 'spawn' method and has to import its target by name.
"""

import os
import time


def succeeds(messages, audio_path, model_size, device):
    messages.put(("status", "Model ready", None))
    messages.put(("progress", 5.0, 10.0))
    messages.put(("result", [(0.0, 1.0, f"{audio_path}|{model_size}|{device}")]))


def reports_pid(messages, audio_path, model_size, device):
    messages.put(("result", [(0.0, 1.0, str(os.getpid()))]))


def fails(messages, audio_path, model_size, device):
    messages.put(("error", "ValueError: bad audio"))


def crashes(messages, audio_path, model_size, device):
    os._exit(3)  # like a segfault in native code: no result, no error message


def runs_forever(messages, audio_path, model_size, device):
    time.sleep(60)
