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
