# SK AI Model Family — Model Card

## 1. Model Family Overview

**SK AI** is a modular, local-first multimodal AI architecture designed as an extensible ecosystem of specialized models rather than a single monolithic black-box network.

- **Developer**: SK AI Architecture & Engineering Team
- **License**: Apache-2.0 (Core Architecture, Local Weights, Tokenizers, Tooling)
- **Primary Languages**: English (`en`), Sinhala (`si` / සිංහල), Sinhala-English code-switched text (`si-en`), and 16+ programming/markup languages.
- **Architecture Family**:
  - `SK-LLM` (Decoder-only Transformer with RMSNorm, RoPE, SwiGLU, Grouped-Query Attention, KV Cache, Weight Tying)
  - `SK-CODE` (Code generation, AST/syntax inspection, build error analysis, multi-language refactoring)
  - `SK-VISION` (Local image signal & OCR feature extractor + VLM multimodal adapter)
  - `SK-IMAGE` (Iterative latent/pixel diffusion & deterministic procedural synthesizer)
  - `SK-MUSIC` (Rights-aware symbolic MIDI + polyphonic multi-track WAV synthesizer & stem separator)
  - `SK-VIDEO` (Temporal frame interpolator & multi-frame video synthesizer with VRAM safety checks)
  - `SK-AUDIO` & `SK-SPEECH` (WAV DSP analyzer, VAD, formant TTS synthesizer, STT pipeline)

## 2. SK-LLM Configurable Tiers

| Model Variant | Layers (`n_layers`) | Hidden Dim (`d_model`) | Heads (`n_heads`) | KV Heads (`n_kv_heads`) | FFN Dim (`intermediate`) | Context Length | Target Parameter Scale |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **SK AI Mini (Local Edge/CPU)** | 4 | 128 | 4 | 2 | 384 | 4,096 | ~2.76M (Installed Local Checkpoint) |
| **SK AI Small** | 24 | 2,048 | 16 | 4 | 5,632 | 8,192 | ~1.3B |
| **SK AI Medium** | 32 | 4,096 | 32 | 8 | 11,008 | 32,768 | ~7.2B |
| **SK AI Large** | 64 | 8,192 | 64 | 8 | 28,672 | 131,072 | ~70.4B |

*Note on Realism*: Only **SK AI Mini** weights (`models/weights/sk_llm_mini_fp32.skbin`), the bilingual Sinhala-English tokenizer (`models/weights/sk_tokenizer_v1.json`), local embedding projections (`models/weights/sk_embed_mini.skbin`), and procedural/DSP media synthesizers are pre-installed in this repository so that the repo stays lightweight while remaining 100% functional out of the box. Multi-billion parameter tiers (`Small`, `Medium`, `Large`) and heavy third-party diffusion/video checkpoints are registered as optional downloadable packages in `models/registry.json` toward the 1 TB modular storage architecture.

## 3. Intended Use & Limitations

- **Intended Use**: Local-first AI research, bilingual Sinhala/English natural language tasks, multi-language software engineering, Android Studio (Kotlin/Jetpack Compose) project generation, Windows `.EXE` desktop project generation, rights-aware music arrangement, local document RAG, and secure agentic tool execution.
- **Limitations**:
  - `SK AI Mini` is a compact local checkpoint designed for instant CPU execution, architectural verification, and deterministic hybrid inference; it does not claim frontier LLM benchmark equivalence without large-scale pretraining compute.
  - High-resolution video generation (`SK-VIDEO` Large) and `SK-LLM Large` require high-VRAM GPUs (24 GB–80 GB+ VRAM). The hardware compatibility subsystem (`hardware/compatibility.py`) proactively warns before attempting to load weights that exceed detected system memory.
  - **Copyright & Voice Cloning Policy**: `SK-MUSIC` and `SK-SPEECH` strictly prohibit cloning living artists' voices or reproducing copyrighted melodies. Generated musical arrangements carry no legal guarantee of copyright-free status and must be reviewed by the user.
