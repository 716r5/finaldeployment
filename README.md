# Hyperspectral Oil-Spill Detection on FPGA

notes 

*1197 have the results in them

*sometimes the fpga doesnt detect the usb, if that happens leave the fpga alone for a while or type dfx-mgr-client -load dpu and then xdputil query 

*kobe is the name of the usb, replace with number of your device while using 

A deployment package for running a trained hyperspectral CNN on the PlanetSpark X7 AI Edge V23. The INT8 model runs on the board's DPU. The CPU prepares patches, applies variance-adaptive thresholding and builds the scene maps.

The input is a stored hyperspectral scene. Each 11 × 11 patch contains 138 spectral bands and produces one centre-pixel prediction. The output includes an oil/water map, an overlay, pixel coordinates and classification metrics when labels are available.

## Requirements

- PlanetSpark X7 AI Edge V23 with its configured Linux image and DPU overlay.
- DPU target: `DPUCZDX8G_ISA1_B4096`; fingerprint `0x101000056010407`.
- Board Python with NumPy, XIR and VART. The saved run used Python 3.10.6, NumPy 1.23.3 and Vitis AI 3.5.
- Root access on the board, `xdputil`, `timeout`, and enough persistent storage for the application, scene data and outputs.
- A USB drive to transfer the package and collect results.

The supplied XModel is compiled for this DPU target. It does not run directly on a Mac or a different FPGA configuration. A laptop can prepare inputs and display the saved reports.

## Download

Clone the repository:

```bash
git clone https://github.com/716r5/finaldeployment.git
cd finaldeployment
```

The model, test patch, labels, scripts and saved results are in the repository. The three prepared scene cubes are in the [deployment-data-v1 release](https://github.com/716r5/finaldeployment/releases/tag/deployment-data-v1). Download `prepared-scenes.tar.gz` and extract it **inside this folder**, preserving its directory structure:

```bash
tar -xzf /path/to/prepared-scenes.tar.gz -C .
```

With GitHub CLI, the download can also be done with:

```bash
gh release download deployment-data-v1 --repo 716r5/finaldeployment --pattern prepared-scenes.tar.gz --dir .
tar -xzf prepared-scenes.tar.gz
```

The cubes total about 1.57 GiB uncompressed. All three are required by the package manifest, although only GM01 and GM02 are in the active queue. GM03 is retained as validation data.

## Check the package

On a laptop with Python 3 and the preparation dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-laptop.txt
python -B deploy.py verify
```

On Windows, `INSTALL_LAPTOP_DEPS.cmd` installs the preparation dependencies. `PREPARE.cmd` verifies the already prepared package.

Verification checks the files against `release.json`; it does not execute the FPGA. Do not regenerate the manifest to bypass a missing-file or checksum error. Restore the matching file instead.

## Run on the board

Copy the complete folder to the USB drive. Find the mount path on the board:

```bash
lsblk -f
```

Run the following, replacing `kobe-sda1` with the actual mount name:

```bash
sudo -E bash /run/media/kobe-sda1/finaldeployment/RUN.sh
```

This verifies and installs the package, checks the DPU, runs the test patch, measures patch latency, processes the scene queue and exports results to the USB. It stops if a stage fails.

The application is installed in `/opt/oil-spill`. Results, checkpoints and logs are stored in `/var/lib/oil-spill`. The default command runs manually and does not enable automatic startup.

### Run options

```bash
sudo -E bash /run/media/kobe-sda1/finaldeployment/RUN.sh run
sudo -E bash /run/media/kobe-sda1/finaldeployment/RUN.sh fresh
sudo bash /run/media/kobe-sda1/finaldeployment/RUN.sh status
```

- `run`: resume interrupted scenes or reuse verified completed maps.
- `fresh`: scan the queued scenes again and create new results.
- `status`: show the saved execution status.

A repeated run can reuse maps. Look for `PROCESSED_ON_FPGA` or `REUSED_COMPLETED_SCENE` in the queue results to distinguish new inference from reuse.

## Read the results

Open `report.html` inside a scene directory under `results-*/scenes/`. Each report includes:

| File | Contents |
| --- | --- |
| `scene_preview.png` | Three-band composite of the input scene |
| `oil_water_map.png` | Predicted oil/water classes |
| `overlay.png` | Predictions over the scene composite |
| `summary.json` | Threshold, timing, class counts and label-based metrics |
| `oil_regions.csv` | Connected oil regions and pixel coordinates |
| `classes.npy` | 0 = water/background, 1 = oil, 255 = unclassified |
| `logits.npy`, `oil_probability.npy` | Model scores at classified centres |

Red marks predicted oil, blue marks predicted water/background, and grey marks unclassified pixels. The outer five-pixel border is unclassified because the model requires a full 11 × 11 patch. Coordinates are zero-based image coordinates, not latitude and longitude.

### Included runs

- `results-20221109T072944-1197`: contains the adaptive GM01/GM02 results in scene folders ending in `-mad1`, plus retained earlier outputs.
- `results-20221109T055324-971`: earlier fixed-threshold run.

The board clock was incorrect. The 2022 folder timestamps do not represent the actual experiment dates.

| Scene | Adaptive threshold | Oil precision | Oil recall | Oil F1 | Scan time |
| --- | ---: | ---: | ---: | ---: | ---: |
| GM01 | 0.767546 | 0.5237 | 0.9009 | 0.6623 | 302.784 s |
| GM02 | 0.990000 | 0.7219 | 0.5425 | 0.6195 | 445.894 s |

These values come from the included adaptive-run summaries. Metrics exclude the five-pixel border. The single-runner benchmark recorded a median model-call latency of 0.212635 ms. Full-scene scan time also includes patch handling and map assembly; it is not the same timing measurement.

## Prepare additional scenes

Inputs must be MATLAB `.mat` files containing `img` in height × width × spectral-band order, with the sensor and band ordering expected by this model. An optional `map` contains labels 0 = water/background and 1 = oil. RGB images and screenshots cannot be used as model inputs.

Preprocessing selects the fixed 138 bands and normalises each band using its scene-level 1st and 99th percentiles. Preparation requires the packages in `requirements-laptop.txt` and Tkinter for the file picker.

On Windows, run `ADD_IMAGES.cmd`. On macOS or Linux, activate the preparation environment and run:

```bash
python prepare_queue.py --additional
```

This creates a queue under `additional_inputs/` without changing the original queue. Copy it to the USB, then import it on the installed board:

```bash
read -r -p 'Path to the additional queue.json: ' INPUT_QUEUE
sudo /bin/bash -c 'source /opt/oil-spill/runtime.env; "$PYTHON_BIN" /opt/oil-spill/queue_app.py import --input-queue "$1"' _ "$INPUT_QUEUE"
sudo /bin/bash -c 'source /opt/oil-spill/runtime.env; exec /bin/bash /opt/oil-spill/RUN.sh run "$PYTHON_BIN"'
```

## Files and documentation

- `deploy.py`: model verification, single-patch execution and benchmarking.
- `scene.py`: full-scene inference, thresholding and report generation.
- `queue_app.py`: scene queues, imports and resume handling.
- `prepare_scene.py`, `prepare_queue.py`: laptop-side input preparation.
- `hsi_contract.py`: spectral-band and preprocessing configuration.
- `regions.py`: connected-region extraction.
- [Board troubleshooting](LAB_GUIDE.md)
- [Validation and result interpretation](VALIDATION.md)

This package performs inference on stored scenes. It does not train the CNN, acquire live sensor data or transmit satellite data.
