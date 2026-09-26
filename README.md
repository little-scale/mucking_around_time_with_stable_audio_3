# SA3 Browser Tools

A browser interface for Stable Audio 3 with interchangeable **CUDA/PyTorch** and **Apple MLX** backends. The model, cache, uploads, and generated audio remain on the machine running the server; other computers on the LAN need only a browser.

## Pages

| URL | Tool |
| --- | --- |
| `/monitor` | Graphical Pipeline Monitor, live sampler events, sweeps, and downloads |
| `/sfx-matrix` | Small SFX prompt matrix and selectable playlist |
| `/same-lab` | SAME latent editing and optional OSC control |
| `/drift` | Three-track Drift Looper |
| `/loop-mutator` | Focused single-loop mutation instrument |
| `/beat-reconstructor` | Four-track event sequencer and loop reconstruction workstation |
| `/live-audio-diffusion` | Non-overlapping, deadline-driven live audio-to-audio diffusion |

Open `/` for the landing page and links to every tool.

## Getting started: RTX Linux server

The tested target is the official Stable Audio 3 `0.1.0` source at commit `124e8a7`, Python 3.10+, PyTorch 2.7.1, and CUDA 12.6 on an RTX 4080.

1. Install the official project and environment:

   ```bash
   git clone https://github.com/Stability-AI/stable-audio-3.git
   cd stable-audio-3
   git checkout 124e8a7
   uv sync
   ```

2. Accept the terms for the required Stability AI models, then authenticate:

   ```bash
   uv tool install huggingface_hub
   hf auth login
   ```

3. Put this folder inside `stable-audio-3`:

   ```text
   stable-audio-3/
   ├── .venv/
   ├── stable_audio_3/
   └── sa3_monitor_browser_tool/
   ```

4. Start the server:

   ```bash
   cd sa3_monitor_browser_tool
   chmod +x sa3-monitor
   ./sa3-monitor --backend cuda
   ```

   The launcher finds the parent `.venv`, checks dependencies, and listens on `0.0.0.0:7861`. The first generation downloads the selected model to the Linux user's cache. The Medium model also requires a `flash-attn` build or wheel compatible with the installed Python, PyTorch, and CUDA versions.

5. Find the Linux LAN address:

   ```bash
   hostname -I
   ```

   On the Mac, open `http://<linux-ip>:7861`. No model or Python installation is needed on the Mac.

Verify CUDA if startup fails:

```bash
../.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Getting started: Apple Silicon / MLX

Install the upstream MLX implementation:

```bash
git clone https://github.com/Stability-AI/stable-audio-3.git
cd stable-audio-3/optimized/mlx
./install.sh
```

Then launch the monitor:

```bash
cd /path/to/sa3_monitor_browser_tool
./sa3-monitor --backend mlx --sa3-root /path/to/stable-audio-3
```

MLX listens on `127.0.0.1:7861` by default. An existing `optimized/mlx/.venv` is selected automatically.

## Launch options

```bash
./sa3-monitor --backend auto
./sa3-monitor --backend cuda
./sa3-monitor --backend mlx
./sa3-monitor --diagnose
```

Useful overrides:

```bash
./sa3-monitor --backend cuda --host 0.0.0.0 --port 7861
./sa3-monitor --backend cuda --sa3-root /path/to/stable-audio-3
./sa3-monitor --backend cuda --output-dir /path/to/output
./sa3-monitor --backend cuda --ssl-certfile /path/to/lan-cert.pem --ssl-keyfile /path/to/lan-key.pem
```

Auto-detection prefers usable CUDA on Linux, then MLX on Apple Silicon. It does not silently fall back to CPU. Backend, device, VRAM, dtype, output path, and startup errors are shown in the terminal and UI.

## Loopers

- **Loop Beats** sets the generated waveform length.
- **Mutation Interval** sets how many complete loops pass before requesting the next version.
- After bootstrap, **Mutation** becomes audio-conditioning `sigma_max`.
- Loop Mutator settings and prompt tags save automatically and apply to the next generation; no Apply button is required.
- Word-cloud tags plus BPM form its positive prompt. Blank hidden role and negative prompts are omitted.
- Clear Output removes generated lineage files while preserving controls.

All interfaces share one accelerator lock. Concurrent generation requests are serialized, with the client IP and prompt printed first in the Linux terminal. Each browser receives the events and audio for its own run ID.

## Live Audio Diffusion

The live page records independent PCM chunks in the browser and sends one chunk
at a time through SA3 audio conditioning. Every dry chunk is scheduled first on
a fixed output timeline. A processed result replaces it when inference finishes
before the playback deadline; late or failed chunks remain dry, so the stream
does not stop or accumulate unbounded latency. Model inputs never overlap.

Microphone and audio-interface capture requires a browser secure context.
`http://127.0.0.1` and `http://localhost` are accepted by browsers, but a page
opened from another computer as `http://<linux-ip>:7861` normally cannot request
an input device. For LAN capture, either use `--ssl-certfile` plus
`--ssl-keyfile` with a certificate trusted by the browser computer, or put SA3
Monitor behind a trusted HTTPS reverse proxy. The certificate must include the
hostname or IP used in the browser. Keep the service private: it has no
authentication.

For one remote browser, an SSH tunnel is the simplest secure option and works
without certificates or internet access. Bind SA3 Monitor locally on the GPU
host:

```bash
./sa3-monitor --backend cuda --host 127.0.0.1 --port 7861
```

Then create the tunnel on the browser computer and open the localhost URL:

```bash
ssh -N -L 7862:127.0.0.1:7861 user@GPU_HOST
```

```text
http://127.0.0.1:7862/live-audio-diffusion
```

The optional boundary fade defaults to off because independent, non-overlapping
chunks would otherwise both attenuate at each join and create a short audible
dip.

The timing controls accept exact millisecond values. Chunk duration and
processing buffer both have a 500 ms minimum. The displayed end-to-end latency
is their sum. Entering a BPM creates clickable 4/4 suggestions for half-beat,
beat, two-beat, bar and multi-bar chunk durations; suggestions outside the
500–10000 ms live range are omitted.

The **Prompt sequence** starts with one prompt. Press **+** to add up to 32
prompt cards. Each consecutive captured chunk uses the next prompt, and the
list loops back to the first prompt at the end. All prompt cards share the
current model, negative prompt, steps, CFG, APG, max sigma, and seed. For
example, selecting a two-bar chunk makes every prompt card last for one
two-bar input block before the next prompt is used.

The inverted prompt card is deliberately driven by the scheduled audio output,
not by request or GPU activity. It therefore shows the processed prompt result
currently being heard. If a result misses its deadline and the dry input block
plays instead, no prompt card is highlighted and the display says **DRY
FALLBACK**. Prompt text can be edited while streaming and affects future
chunks; adding and removing prompts is locked until stopped.

### Max / Max for Live with `jweb~`

Load the page in a `jweb~` object rather than `jweb`. The page's Web Audio mix
is connected to `AudioContext.destination`, so `jweb~` exposes it through its
two MSP signal outlets. Connect those outlets to a stereo `gain~` / `dac~` in
Max, or to the left and right device outputs in Max for Live.

```text
jweb~ @url http://127.0.0.1:7862/live-audio-diffusion
```

The page uses the `window.max` bridge injected by `jweb~` (this is distinct
from the Node for Max `require("max-api")` module). Send these messages to the
`jweb~` inlet while the stream is stopped:

```text
stream start
stream stop
bpm 120
chunk 1000
latency 1000
```

`chunk` and `latency` are milliseconds and clamp to their supported ranges.
The page emits `live_diffusion ready`, `stream start`, `stream stop`,
`stream error <reason>`, and `prompt hearing <number> <text>` through the Max
message bridge. A dry fallback emits `prompt hearing 0 dry`. Max handles these
messages asynchronously on its low-priority queue; audio itself remains on the
two signal outlets. See Cycling '74's
[Web Browser and jweb guide](https://docs.cycling74.com/userguide/web_browser/)
and [`jweb~` reference](https://docs.cycling74.com/reference/jweb~/).

## SAME Lab and OSC

OSC is off by default. To enable it on a trusted LAN:

```bash
./sa3-monitor --backend cuda --osc --osc-host 0.0.0.0 --osc-port 9000
```

Arm the intended SAME Lab browser session before sending OSC. Only one session is armed at a time.

## Files and LAN safety

Generated files default to `output/` on the server host. Models use the host user's normal Hugging Face cache. Neither is included in the project ZIP.

The server has **no authentication**. Do not port-forward port 7861 or expose it through a public tunnel. Restrict access to the trusted LAN, for example:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 7861 proto tcp
```

Replace the subnet with your own. Use an authenticated TLS reverse proxy for any non-LAN deployment.

## Tests

The test suite uses mocks and does not require CUDA or MLX hardware:

```bash
uv pip install -r requirements-test.txt
uv run pytest -q
```

## Troubleshooting

- **CUDA unavailable:** run `nvidia-smi`, check the selected Python, and verify matching PyTorch/CUDA packages.
- **Medium fails or produces glitches:** check that `flash-attn` matches Python, PyTorch, CUDA, and the RTX 4080 architecture.
- **Model 401/403:** accept the model terms and run `hf auth login` as the server user.
- **Mac cannot connect:** use the Linux LAN IP, confirm the server says `0.0.0.0:7861`, and check the firewall.
- **Port busy:** stop the earlier process or launch with `--port 7870`.
- **MLX source missing:** pass `--sa3-root` or set `SA3_MLX_ROOT` to the upstream `optimized/mlx` directory.
- **A request is waiting:** another page or classroom client is using the shared accelerator.

## Architecture and protocol

FastAPI serves every frontend and a shared protocol containing stage, tensor preview, sampler-step, metrics, audio-ready, completion, and error events. `backends/base.py` defines the stable engine interface; `backends/cuda.py` and `backends/mlx.py` translate native progress into that contract.

To add another backend, implement the base interface, register it in `backends/factory.py`, and add mocked contract tests. Frontend code does not need to change.

## Upstream and license

- Upstream: [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3), package `0.1.0`, commit `124e8a7`.
- This project does not bundle upstream code, model weights, or caches.
- This project is released under the [0BSD license](LICENSE).
- Upstream source, model weights, and model outputs remain subject to their own licenses and terms.
