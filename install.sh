#!/bin/bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo 'Using RUN.sh: sequential checks and real inference, stopping on the first failure.'
echo 'Automatic startup is optional; run RUN.sh autostart separately only if wanted.'
exec bash "$HERE/RUN.sh" run "${1:-/usr/bin/python3}"
