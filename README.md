# SK AI — Modular Local-First Multimodal AI Ecosystem

**SK AI** is an extensible, local-first multimodal AI platform and model family architected for **Text, Reasoning, Bilingual Sinhala/English Understanding, 17-Language Software Engineering, Android Studio (Kotlin/Jetpack Compose) App Generation, Windows `.EXE` Desktop Generation, Rights-Aware Music Composition & Arrangement (MIDI/WAV/Stems), Iterative Image Synthesis, Temporal Video Frame Synthesis, Audio/Speech Processing, Document RAG, and Permission-Gated Agentic Tool Execution**.

---

## 1. What SK AI Is (and Realism Principles)

SK AI is designed as a **modular model family** rather than a single monolithic neural network or a thin wrapper around external proprietary cloud APIs.

```text
                    SK AI CORE
                        |
        +---------------+---------------+
        |               |               |
     SK-LLM          SK-VISION       SK-CODE
        |               |               |
     Text/Reasoning   Image/VLM      Programming
        |
   SK-MULTIMODAL ORCHESTRATOR
        |
  +-----+------+-------+-------+-------+
  |            |       |       |       |
 MUSIC       IMAGE    VIDEO   AUDIO   SPEECH
  |
 SK TOOL/AGENT SYSTEM
  |
 +-----------------------------+
 | Android | Windows | Web | CLI |
 +-----------------------------+
```

- **Local-First & Independent**: SK AI implements its own decoder-only Transformer (`SKLLMForCausalLM`) with **RMSNorm**, **Rotary Positional Embeddings (RoPE)**, **SwiGLU**, **Causal Grouped-Query Attention (GQA)**, **KV Caching**, and **Weight Tying**, alongside a lossless **Sinhala + English + Code Tokenizer** (`SKTokenizer`).
- **No Fabricated Frontier Claims**: Out of the box, SK AI initializes and runs **SK AI Mini** (`2,756,736` parameters, `FP32` weights verified via SHA-256 on disk) and local procedural/DSP synthesizers on any CPU or GPU. Multi-billion parameter configurations (`SK AI Small 1.3B`, `SK AI Medium 7.2B`, `SK AI Large 70.4B`) and heavy third-party diffusion/video/ASR weights (`SDXL`, `Open-DiT 14B`, `Whisper Large-v3`) are supported via clean adapter interfaces and reported honestly as `NOT INSTALLED` until downloaded.

---

## 2. What SK AI Can Actually Do Out of the Box vs. Downloaded Models

| Capability | Out-of-the-Box Local Engine (`BUILT BY SK AI`) | Optional Heavy Checkpoints (`NOT INSTALLED` by Default) | Requires GPU? | Runs Offline (`offline_mode=true`)? |
| :--- | :--- | :--- | :--- | :--- |
| **Bilingual Chat & Reasoning (Sinhala & English)** | `SK AI Mini` (4L, 128D, GQA, KV Cache) + Local Knowledge Synthesis | `SK AI Small (1.3B)`, `Medium (7.2B)`, `Large (70.4B)` | Mini: No (CPU) / Medium & Large: Yes | Yes |
| **Coding AI (17 Languages + AST & Build Analysis)** | `SK-CODE` Multi-language generator, Python AST debugger, unified diff patcher, Gradle/TS build analyzer | `SK-CODE 7B/34B` checkpoints | No | Yes |
| **Android Studio Project Generation** | Generates full Kotlin + Jetpack Compose + Gradle KTS + AndroidManifest + Dual-Mode Local/LAN client | Optional local mobile NPU GGUF weights | No | Yes |
| **Windows `.EXE` Project Generation** | Generates PyInstaller `.spec`, Electron, or Tauri project + Windows build scripts | Requires Windows host OS toolchain to compile native PE32+ `.exe` | No | Yes |
| **Music Generation & Arrangement (`SK-MUSIC`)** | Symbolic composer, Sinhala pop & cinematic arranger, SMF `.mid` generator, polyphonic `.wav` synth, 4-stem exporter | Optional neural audio codec weights | No | Yes |
| **Image Generation (`SK-IMAGE`)** | Seeded iterative latent/pixel denoiser + native RFC-2083 PNG encoder | `SDXL 1.0 Base` (`6.94 GB`) | Local: No / SDXL: Yes (12 GB+ VRAM) | Yes |
| **Video Generation (`SK-VIDEO`)** | Temporal motion trajectory interpolator + multi-frame PNG sequence exporter + VRAM pre-flight checker | `Open-DiT 14B Video` (`28.0 GB`) | Local: No / DiT: Yes (24 GB+ VRAM) | Yes |
| **Vision & Document RAG (`SK-VISION`, `documents/`, `memory/`)** | Binary PNG/JPEG/PPM/BMP parser, VLM MLP projector, DOCX/XLSX/PDF/MD extractor, 64-dim bilingual vector store | Optional high-res OCR/VLM checkpoints | No | Yes |

---

## 3. Installation & Quick Start

### Python & CLI Usage

```bash
# 1. Run the 16-point subsystem smoke test
python3 scripts/smoke_test.py

# 2. Inspect actual installed model sizes, SHA-256 checksums, and 1 TB library budget
python3 scripts/storage_report.py

# 3. Run the automated test suite
pytest

# 4. Use the SK AI CLI
python3 -m skai.cli chat "Explain quantum computing in Sinhala."
python3 -m skai.cli code "Write a Rust tensor stats struct"
python3 -m skai.cli music "Create a happy Sinhala pop song"
python3 -m skai.cli image "Emerald tea hills in Sri Lanka at sunrise"
python3 -m skai.cli video "Ocean waves at sunset"
python3 -m skai.cli android "Create a Sinhala AI chat app."
python3 -m skai.cli windows "Make this HTML app into a Windows EXE."
python3 -m skai.cli models list
python3 -m skai.cli hardware
python3 -m skai.cli benchmark
```

### Unified Python API

```python
from skai import SKAI

ai = SKAI(offline_mode=True)

# 1. Bilingual Chat (Sinhala + English)
res = ai.chat("Explain quantum computing in Sinhala.")
print(res["answer"])

# 2. Code Generation
code = ai.generate_code("Create a Kotlin Jetpack Compose card", language="Kotlin")
print(code["code"])

# 3. Music Generation (Exports .mid, .wav, and 4 stems)
music = ai.generate_music("Create a happy Sinhala pop song", duration_sec=3.5)
print(music["midi_path"], music["wav_path"], music["stems"])

# 4. Image & Video Generation
img = ai.generate_image("Futuristic library in Colombo", width=160, height=160, seed=42)
vid = ai.generate_video("Sunrise over mountains", duration_sec=1.5, fps=4)

# 5. Android Studio & Windows EXE Project Generation
android_proj = ai.create_android_project("Create a Sinhala AI chat app.")
windows_proj = ai.create_windows_project("Make this HTML app into a Windows EXE.")
```

### Starting the Full-Stack Local Web Workbench

```bash
npm run dev
# Opens the SK AI Workbench on http://localhost:3000 connected to the local Python SK AI engine
```

---

## 4. Training, Quantization & Adding Models

- **Pretraining & SFT**: Configure `configs/training_config.json` and run `PretrainingPipeline` (`training/pretrain.py`), `SFTPipeline` (`training/sft.py`), or `MultimodalAdapterTrainer` (`training/multimodal_adapter.py`).
- **Dataset Validation**: Use `datasets/manager.py` (`SKDatasetManager`) to deduplicate records via SHA-256, detect Sinhala (`si`), English (`en`), and mixed (`si-en`) scripts, and enforce open-license filtering (`Apache-2.0`, `MIT`, `CC-BY-4.0`).
- **Quantization**: Use `quantization/quantizer.py` (`SKQuantizer`) to quantize weights across `FP32`, `FP16`, `BF16`, `INT8`, and `4-bit` and measure actual reconstruction MSE and SNR (dB).
- **Adding Models to the 1 TB Library**: Place `.skbin` or `.safetensors` weights in `models/weights/` and register their metadata in `models/registry.json`. `SKModelManager` automatically computes and verifies SHA-256 hashes and real byte sizes on disk.

---

## 5. Hardware, Storage, Privacy & Copyright Policies

- **Hardware Requirements**:
  - **Minimum (SK AI Mini + Procedural Studio)**: 2 CPU cores, 1 GB RAM, 50 MB disk space.
  - **Recommended for Full 1 TB Ecosystem (Small/Medium/Large + SDXL + Open-DiT Video)**: 16+ CPU cores, 64 GB System RAM, NVIDIA GPU with 24 GB–80 GB VRAM, 1 TB NVMe SSD.
- **Privacy (`offline_mode=true`)**: Enabled by default in `configs/system_config.json`. Blocks all external network/API requests and telemetry. Local vector memory (`memory/`) is 100% controllable and erasable by the user.
- **Copyright & Anti-Voice-Cloning Policy**: `SK-MUSIC` and `SK-SPEECH` refuse prompts attempting to clone living artists' voices or reproduce copyrighted songs. Generated musical arrangements carry no legal guarantee of copyright-free status.
- **License**: Apache License 2.0 (see `LICENSE`).
