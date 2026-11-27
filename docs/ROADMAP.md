# Roadmap and open problems

Updated 27 November 2026. Each problem lists what is already known, so you don't have to rediscover it. The deep detail, with addresses and every diagnostic switch, is in [native-host.md](native-host.md).

**Where it stands:** On the maintainer's Mac, story mode reaches the prologue world and dialogue plays. The picture is still mostly black and world frames are slow. [Maintainer's report on #15](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/15#issuecomment-7205388027).

<a id="1-the-prologue-crash-top-priority-15"></a>

## 1. Prologue status ([#15](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/15))

The GPU memory overwrite is fixed on main. The ray-tracing builder asked for 45 threads. The host launched 45 groups of 64 and wrote empty nodes over neighbouring resources. The Metal host now rounds the thread count up to whole groups and masks the surplus lanes. The [dispatch investigation](native-host.md#dispatchx-with-thread-counts) records the evidence and regression tests.

A cold first world frame can still trip the title's 30-second watchdog during pipeline compilation. For a new report on #15, include the commit, console model, cache state and run log. Use `tools/macos/run-summary.sh <log>` to find the last recorded frame and any exception. Rendering, texture copies, frame time, shader caching and compiler failures are tracked separately in [#182](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/182), [#178](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/178), [#179](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/179), [#180](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/180) and [#181](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/181).

## 2. Frame rate and memory in the world

- In the world, fps falls from about 15 to about 7 over 4 minutes. Something grows every frame; host texture copies are the main suspect (one tester counted 26,000 by frame 4096). Being investigated on `fix/fps-decay`.
- Series X can run out of GPU memory at the world load, even on 24 GB ([#8](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/8), [#79](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/79), [#110](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/110)). A byte budget on host texture copies is proposed in #110. Series S is the default on 16 GB Macs. The native host now logs Metal's own recommended working set size alongside the title's declared budget at startup, so a tester's log carries the numbers needed to set a real threshold.
- The game rewrites about 40,000 descriptor records a frame with plain stores. Patching its store sites instead of remirroring is being measured ([#43](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/43); `VICTORY_TRACE_DESCRIPTORS=sites`).
- Textures the GPU writes and the CPU reads back (the water simulation) are copied back behind `VICTORY_TEXTURE_BACK=1`, off by default ([#46](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/46)). A world run on an M5 Pro names 22 textures at no measurable cost, then crashes before boats can be seen. Needed: a run past that crash.

## 3. Shader translation gaps ([#4](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/4))

Refused shaders drop their draws and dispatches. Unique refused shaders in the story load fell from 526 to about 150 on 27 November, and more fixes have landed since. Refusals now name their exact cause. The largest groups left:
- a descriptor loaded from title memory at an address in a root constant (bindless; needs the host to map the address to a heap index);
- a lane-mask join of an untracked mask;
- cross-lane operations and LDS shared between the two host waves of a console wave, inside real per-lane branches (needs if-conversion so both host waves always arrive);
- a whole-wave section step the simulation does not follow.

Fixed on 27 November: compute root constants, sized from the root signature's `Num32BitValues` (#114); `v_pk_fmac_f16` constants; wave64 lane-mask joins; `s_absdiff_i32`; signed BC6H; ExecuteIndirect with a count buffer. A batch on `fix/shader-refusals-2` also covers samplers built from constant bit masks, the thread-group info SGPR, and LDS behind uniform branches; it is being merged.

Since then: image stores of some channels (#94) and whole-wave float products (`WaveActiveProduct`, `v_mul_f32`).

Translated shaders, pipelines and refusals are cached on disk between runs.

**Start here:** `crates/victory-shader/src/emit/`. Tests use hand-assembled GFX10 code (`gfx_refusal_tests.rs`), so a refusal shape can be fixed without the game.

## 4. Running without Rosetta (arm64 recompilation)

The whole game executable (about 440,000 functions) lifts to C in seconds and compiles to arm64 (about 80 minutes at 6 jobs). A contributor's branch ([#115](https://github.com/LakshanSivakumar/Native-Victory-macOS/pull/115)) boots it to the intro at 30 fps. It is being tested here, and it is 85 commits behind main.

## 5. Everything after that

- Reverb: the audio DSP's convolution has a synthetic CPU fallback behind `VICTORY_GUESS_DSP=1` (a fixed decaying-noise tail, not the title's own impulse response); default behaviour is unchanged (silence) until a real run judges whether it sounds right ([#17](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/17)). The hardware's impulse-response layout is still undocumented; `VICTORY_TRACE_DSP=1` logs the title's own `ir` data to help settle it.
- The Brightness logo draws as half a rectangle ([#16](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/16)).
- Mesh-shader pipelines: indirect mesh dispatches still drop.
- Culling, depth, stencil and write mask in guessed pipelines are still partly guessed.

## Solved recently

- **26–27 November:**
  - The Metal GPU timeout at about 53 s ([#2](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/2)): waits stay on the CPU until the GPU work is observed done.
  - Failed command buffers now still signal their fences ([#80](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/80)).
  - The black window on M1/M2 ([#26](https://github.com/LakshanSivakumar/Native-Victory-macOS/issues/26), #91).
  - The boot movie's green frames and stutter.
  - Pool pages owned twice after `VirtualFree` (#108); physical page numbers reused (#124); decommit keeps page views (#126).
- **Earlier:**
  - Black content draws.
  - The present pass.
  - Boxed glyphs (#6).
  - The "press A" start screen, now with keyboard and controller input.
  - The story-load deadlock.

## Good first contributions

- Run the game on your Mac and file a tester report; different chips and RAM sizes help, especially runs that reach the prologue.
- Pick a refused shader shape from section 3. It's self-contained and testable without the game.
- Turn a finding in native-host.md into a unit test.
- Improve `tools/macos/native-setup.sh` and `run-native.sh` for setups we haven't seen.
