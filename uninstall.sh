#!/bin/bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then echo 'Run with sudo bash uninstall.sh'; exit 1; fi
if systemctl list-unit-files oil-spill.service --no-legend | grep -q '^oil-spill.service'; then
  systemctl disable --now oil-spill.service
fi
if [ -f /etc/systemd/system/oil-spill.service ]; then
  if ! grep -q 'PlanetSpark hyperspectral CNN boot prediction' /etc/systemd/system/oil-spill.service; then
    echo 'Unrecognized service; refusing to remove it.'; exit 1
  fi
  rm -- /etc/systemd/system/oil-spill.service
fi
systemctl daemon-reload
if [ -f /etc/systemd/system/oil-spill.service.d/20-dpu-manager.conf ]; then
  if grep -q '^# finaldeployment detected DPU-manager ordering' /etc/systemd/system/oil-spill.service.d/20-dpu-manager.conf; then
    rm -- /etc/systemd/system/oil-spill.service.d/20-dpu-manager.conf
    rmdir -- /etc/systemd/system/oil-spill.service.d 2>/dev/null || true
  fi
fi
systemctl daemon-reload
systemctl reset-failed oil-spill.service 2>/dev/null || true
echo 'Automatic startup removed. Application, reference and results preserved.'
echo 'For reinstall, archive and rename /opt/oil-spill and /var/lib/oil-spill as described in LAB_GUIDE.md.'
