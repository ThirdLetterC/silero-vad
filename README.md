# `silero-vad`

Real-time Silero VAD demo in strict C23 with ONNX Runtime C API.

## Requirements
- Zig 0.15+ (for the build runner)
- ONNX Runtime development files (headers + shared library)
- C toolchain capable of `-std=c23`

## Build
```sh
zig build \
  -Dort-include=/path/to/onnxruntime/include \
  -Dort-lib=/path/to/onnxruntime/lib
```
Flags default to system paths if not provided.

## Run
Place input audio at `test.wav` (16 kHz expected). Then:
```sh
zig build run
```
Detected speech segments are printed and saved as `audio/segment_<n>.wav`.

## Download model

```
wget https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx
```

## ONNX Runtime in the project folder
If you prefer a local copy, download the ONNX Runtime release archive (CPU build) into the repo root and unpack it, then point the build at it:
```sh
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar xf onnxruntime-linux-x64-1.18.0.tgz
zig build \
  -Dort-include=onnxruntime-linux-x64-1.18.0/include \
  -Dort-lib=onnxruntime-linux-x64-1.18.0/lib
```

## Project layout
- `src/include/`: public headers (`silero_vad.h`, `wav.h`)
- `src/`: library sources and CLI (`silero_vad.c`, `wav.c`, `main.c`)
- `build.zig`: Zig-based build script targeting `-std=c23` with strict warnings
- `justfile`: convenience tasks (`just install`, `just run`, `just fmt`)

## License
MIT. See `LICENSE` for details.
