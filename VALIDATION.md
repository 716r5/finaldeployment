# Validation

## Package integrity

`release.json` records SHA-256 hashes for the deployment scripts, compiled model, test patch and prepared inputs. After extracting the scene-data release, run:

```bash
python -B deploy.py verify
```

This checks the packaged files. It does not execute the DPU. The Markdown documentation is not part of the sealed runtime hashes.

## Saved hardware results

The adaptive run is stored in `results-20221109T072944-1197`. Its `queue-latest.json` records the execution status, `benchmark.json` contains the patch benchmark, and scene directories ending in `-mad1` contain the adaptive outputs.

The earlier `results-20221109T055324-971` folder contains the fixed-threshold run. Keep its metrics separate from the adaptive results. Board timestamps show 2022 because the clock was incorrect.

## Classification rule

The board CPU calculates scene intensity as the mean over the 138 normalised bands:

```text
MAD = median(abs(intensity - median(intensity)))
robust_std = 1.4826 * MAD
threshold = clip(3.0873 * robust_std + 0.4535, 0.50, 0.99)
predicted_oil = oil_probability > threshold
```

Ground-truth labels are used for evaluation, not for assigning predictions. The model and DPU graph are unchanged by this threshold calculation.

## Evaluation region and data split

GM01 and GM02 are the active evaluation scenes. GM03 is validation-only and remains in the packaged inputs because it is listed in the manifest. It is not in the active queue.

The reported metrics use valid patch centres, excluding a five-pixel border. Oil is the positive class. Precision, recall, F1, IoU and confusion counts are recorded in each scene summary.

The deployment records `NOT_CONFIRMED_HELD_OUT`: it cannot audit the model's training history. A comparison with software results using different border handling is not a pixel-matched measurement of quantisation loss.

## Timing

The saved benchmark uses 20 warm-up calls and 200 measured patch predictions with one runner. Patch model-call latency and full-scene throughput have different measurement boundaries. Scan timing excludes offline laptop preparation and report rendering; see each summary's `timing_scope` field.

## Local preparation records

`LOCAL_INTEGRATION.json` and `REAL_DATA_VALIDATION.json` preserve earlier laptop-side checks. They describe preparation and file handling, not DPU execution. Hardware execution is documented by the saved board exports.

The included results do not establish power consumption, downlink savings or operation aboard a satellite.
