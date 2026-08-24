"""ONE-api connection handling.

Credentials are never stored in this repository. Authentication comes from the
ONE parameter file written by `ONE.setup()`, or from environment variables, and a
missing or broken setup raises immediately instead of dropping into ONE's
interactive prompt (which would hang a non-interactive extraction run).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from . import config


class ONECredentialsError(RuntimeError):
    """Raised when ONE cannot authenticate non-interactively."""


def _setup_hint(base_url: str) -> str:
    return (
        f"Could not connect to {base_url} without prompting.\n"
        "Authenticate once, then re-run:\n"
        "  python -c \"from one.api import ONE; ONE.setup(base_url='"
        f"{base_url}', make_default=True)\"\n"
        "or export IBL_ONE_PASSWORD (and optionally IBL_ONE_USER, "
        "IBL_ONE_BASE_URL, IBL_ONE_CACHE) before running.\n"
        "The IBL public mirror publishes read-only credentials in its own "
        "documentation: https://int-brain-lab.github.io/ONE/notebooks/public_one.html"
    )


@lru_cache(maxsize=4)
def get_one(base_url: str | None = None) -> Any:
    """Return a connected ONE instance, or raise with actionable instructions.

    `silent=True` is what makes this safe for batch runs: it stops ONE from
    prompting on stdin, so a missing setup surfaces as an exception here rather
    than as a stalled job.
    """
    base_url = base_url or os.environ.get("IBL_ONE_BASE_URL") or config.PUBLIC_ONE_BASE_URL

    from one.api import ONE  # imported lazily so `--help` works without ibllib

    kwargs: dict[str, Any] = {"base_url": base_url, "silent": True}
    if os.environ.get("IBL_ONE_PASSWORD"):
        kwargs["password"] = os.environ["IBL_ONE_PASSWORD"]
    if os.environ.get("IBL_ONE_USER"):
        kwargs["username"] = os.environ["IBL_ONE_USER"]
    if config.ONE_CACHE_DIR:
        kwargs["cache_dir"] = config.ONE_CACHE_DIR

    try:
        one = ONE(**kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with a usable message
        raise ONECredentialsError(_setup_hint(base_url)) from exc

    if getattr(one, "alyx", None) is None:
        raise ONECredentialsError(_setup_hint(base_url))
    return one
