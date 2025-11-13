from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from common import run_cli  # type: ignore[import]


ENCODER_NAME = "efficientnet-b7"
DEFAULT_OUTPUT = Path(__file__).with_name("unet_efficientnet_b7_weights.pth")


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    run_cli(
        encoder_name=ENCODER_NAME,
        default_output_name=str(DEFAULT_OUTPUT),
        cli_args=cli_args,
    )


if __name__ == "__main__":
    main()

