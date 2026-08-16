# Copyright 2025 The Anthropomorphism Benchmark Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small shared I/O helpers used by both generator.py and rating.py for
incremental/checkpoint saving during a long-running dialogue generation
or rating session."""

import os
import tempfile
from typing import Any


def atomic_to_csv(df: "Any", path: str, **to_csv_kwargs: Any) -> None:
    """
    Write a DataFrame to `path` atomically.

    Writes to a temp file in the SAME directory as `path` (same
    filesystem is required for os.replace() to be atomic -- a
    cross-filesystem rename is not) then os.replace()s it onto the real
    path. This means the file at `path` is always either the complete
    PREVIOUS version or the complete NEW version, never a truncated/
    half-written one -- even if the process is killed mid-write.

    This matters specifically for incremental/checkpoint saving, where a
    save can happen many times over a long-running session: a plain
    df.to_csv(path) writes directly to the target file, so a crash
    partway through that write corrupts the checkpoint you were trying
    to protect, which is worse than not checkpointing at all (you'd lose
    the previous good checkpoint too, not just the newest one).

    Any kwargs (encoding=, index=, etc.) are passed straight through to
    DataFrame.to_csv().
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=directory, suffix=".tmp", prefix=os.path.basename(path) + "."
    )
    os.close(fd)
    try:
        df.to_csv(tmp_path, **to_csv_kwargs)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
