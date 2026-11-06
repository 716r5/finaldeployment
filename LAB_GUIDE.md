# Board Troubleshooting

Run the setup steps in [README.md](README.md) first. Keep the model, scripts, manifest and prepared scenes together.

## Find the failing stage

`RUN.sh` stops at the first error and prints the stage, command and exit code. Logs are stored under `/var/lib/oil-spill/runs/`.

| Stage | Check |
| --- | --- |
| Runtime/storage | Python can import NumPy, XIR and VART; root access and persistent free space are available. |
| Package verification | Restore any missing or changed file from the matching repository/release. |
| Installation | The existing installation must use the matching model and input identities. |
| DPU readiness | Check the loaded overlay and `xdputil query` output. An empty kernel list is not a ready DPU. |
| Sample inference | Inspect the tensor, fingerprint, label or repeatability error in the log. |
| Benchmark | The sample must pass before benchmark measurements are accepted. |
| Scene processing | Check the scene name, row progress and `queue-latest.json`. Rerun to resume checkpoints. |
| Automatic startup | Check the saved runtime environment and systemd journal. |
| Export | Check USB space and connection. Board results remain available if export fails. |

A `[working]` message means the process is alive. Row counters show scene-processing progress. Keep the USB connected until export finishes.

## Export saved results again

Use a new destination directory and the actual USB mount path:

```bash
sudo bash /opt/oil-spill/RUN.sh export /run/media/kobe-sda1/finaldeployment/results-recovery1
```

## Run a short hardware check

After installation:

```bash
sudo /bin/bash -c 'source /opt/oil-spill/runtime.env; "$PYTHON_BIN" /opt/oil-spill/deploy.py run --load-dpu'
sudo /bin/bash -c 'source /opt/oil-spill/runtime.env; "$PYTHON_BIN" /opt/oil-spill/deploy.py benchmark --load-dpu --warmup 20 --iterations 200'
```

The test patch checks execution and agreement with its known label. It is not an independent accuracy test.

## Optional automatic startup

Manual operation is the default. To enable startup after the manual workflow passes:

```bash
sudo -E bash /run/media/kobe-sda1/finaldeployment/RUN.sh autostart
```

After a clean restart, inspect the service without launching inference manually:

```bash
systemctl is-enabled oil-spill.service
systemctl status oil-spill.service --no-pager
sudo journalctl -u oil-spill.service -b --no-pager
sudo bash /opt/oil-spill/RUN.sh status
```

Check that the sample and queue records belong to the current boot. Previously completed maps may be reused; the sample executes again. A manual run alone does not establish automatic-startup behaviour.

## Model graph and runtime

The compiled graph contains a USER input, one DPU subgraph and a terminal CPU `fix2float` conversion. The runner performs the checked signed-INT8 output conversion on the CPU. This handling is restricted to the supplied model hash and expected tensor scale. Unsupported graphs and fingerprint mismatches are rejected.

The board clock may be incorrect. Use boot IDs to associate runs and the recorded durations for performance measurements.
