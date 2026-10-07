# SK AI — Development & Scaling Roadmap

## Completed Phases (v1.0.0)

- **Phase 1 — SK AI Core**: Decoder-only Transformer (`SKLLMForCausalLM`) with RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU MLP, Grouped-Query Attention (GQA), KV Caching, and weight tying; bilingual Sinhala/English/Code tokenizer; autoregressive inference engine; unified CLI (`skai`).
- **Phase 2 — Coding AI & Agent System**: 16-language coding subsystem (`coding/`), DAG task planner (`agents/planner.py`), tool router, executor, verifier, sandboxed filesystem/shell/git/python tools with explicit permission gates.
- **Phase 3 — Vision, Document AI, Memory & RAG**: Local image parser & screenshot error analyzer (`vision/`), multi-format document parser (`documents/`), local vector store & bilingual embeddings (`memory/`).
- **Phase 4 — Image Generation (`SK-IMAGE`)**: Text-to-image and image-to-image conditioning, iterative seeded sampler, local PNG/PPM output pipeline.
- **Phase 5 — Audio & Speech (`SK-AUDIO`, `SK-SPEECH`)**: WAV signal analysis, Voice Activity Detection (VAD), formant speech synthesis, and transcription adapter.
- **Phase 6 — Music Generation (`SK-MUSIC`)**: Melody/harmony composer, Sinhala pop & cinematic arranger, Song-to-Arrangement transformer, multi-track SMF MIDI generator, polyphonic WAV audio synthesizer, stem exporter, and strict anti-voice-cloning guardrails.
- **Phase 7 — Video Generation (`SK-VIDEO`)**: Text-to-video, image-to-video, and video-to-video pipelines with temporal consistency interpolation, animated GIF/AVI export, and pre-flight VRAM hardware warnings.
- **Phase 8 — Android Integration**: Complete Android Studio project generator (Kotlin, Jetpack Compose, Gradle KTS, AndroidManifest, Sinhala/English resources, Dual-Mode Local Phone / Windows PC LAN client).
- **Phase 9 — Windows EXE Integration**: Multi-stack Windows desktop project generator (PyInstaller, Electron, Tauri, C#/.NET, C++) with build scripts and environment validation.
- **Phase 10 — Unified Web & Desktop Workbench**: Full-stack local-first Web UI & Express/Python bridge with real-time hardware monitoring, 1 TB storage budget inspector, interactive multimodal studio, and automated test suite.

## Future Milestones (v1.1 – v2.0)

1. **Distributed Multi-Node Pretraining**: FSDP & DeepSpeed ZeRO-3 scaling scripts for training `SK-LLM Small (1.3B)` and `SK-LLM Medium (7.2B)` on curated Sinhala + English + Code corpora.
2. **Native Vulkan / Metal Quantized Kernels**: GGUF/AWQ 4-bit kernel bindings for mobile NPU execution on Android ARM64 devices.
3. **3D & Spatial Asset Pipeline**: Extending `SK-IMAGE` and `SK-VIDEO` with multi-view mesh reconstruction.
