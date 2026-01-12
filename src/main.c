#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdckdint.h>

#include "silero_vad.h"
#include "wav.h"

[[nodiscard]]
static bool write_segment(const wav_reader_t* reader, timestamp_t ts, size_t index, const char* directory) {
    if (reader == nullptr || reader->data == nullptr || directory == nullptr) {
        return false;
    }

    if (ts.start < 0 || ts.end <= ts.start) {
        return false;
    }

    size_t start = (size_t)ts.start;
    size_t end = (size_t)ts.end;

    if (start >= reader->num_samples) {
        return false;
    }
    if (end > reader->num_samples) {
        end = reader->num_samples;
    }

    const size_t frames = end - start;
    if (frames == 0U) {
        return false;
    }

    size_t total_samples = 0;
    if (ckd_mul(&total_samples, frames, (size_t)reader->num_channel)) {
        return false;
    }

    const float* segment_data = reader->data + start * (size_t)reader->num_channel;

    char filename[128];
    int written = snprintf(filename, sizeof(filename), "%s/segment_%zu.wav", directory, index);
    if (written <= 0 || written >= (int)sizeof(filename)) {
        return false;
    }

    wav_writer_t writer;
    wav_writer_init(&writer, segment_data, frames, reader->num_channel, reader->sample_rate, reader->bits_per_sample);
    if (!wav_writer_write(&writer, filename)) {
        fprintf(stderr, "Failed to write segment %zu to %s\n", index, filename);
        return false;
    }

    return true;
}

int main() {
    // 1. Read WAV
    constexpr char input_file[] = "test.wav";
    wav_reader_t reader;
    
    printf("Loading WAV file: %s\n", input_file);
    if (!wav_reader_open(&reader, input_file)) {
        return EXIT_FAILURE;
    }

    // 2. Init VAD
    constexpr char model_path[] = "silero_vad.onnx";
    vad_iterator_t vad;
    
    // Default params matching C++ constructor defaults
    printf("Initializing VAD with model: %s\n", model_path);
    const int sample_rate = reader.sample_rate;
    if (sample_rate != 8'000 && sample_rate != 16'000) {
        fprintf(stderr, "Unsupported sample rate: %d (expected 8'000 or 16'000)\n", sample_rate);
        wav_reader_close(&reader);
        return EXIT_FAILURE;
    }
    constexpr int window_ms = 32;
    if (!vad_iterator_init(&vad, model_path, sample_rate, window_ms, 0.5f, 100, 30, 250, INFINITY)) {
        fprintf(stderr, "Failed to initialize VAD\n");
        wav_reader_close(&reader);
        return EXIT_FAILURE;
    }

    // 3. Process
    printf("Processing %zu samples...\n", reader.num_samples);
    vad_iterator_process(&vad, reader.data, reader.num_samples);

    // 4. Output Results
    const float sample_rate_float = (float)sample_rate;
    constexpr char output_directory[] = "audio";
    size_t segment_index = 0;

    for (size_t i = 0; i < vad.speeches.size; i++) {
        const auto ts = vad.speeches.data[i];
        const auto start_sec = rintf((ts.start / sample_rate_float) * 10.0f) / 10.0f;
        const auto end_sec = rintf((ts.end / sample_rate_float) * 10.0f) / 10.0f;
        
        printf("Speech detected from %.1f s to %.1f s\n", start_sec, end_sec);

        if (write_segment(&reader, ts, segment_index, output_directory)) {
            printf("  -> Saved segment to %s/segment_%zu.wav\n", output_directory, segment_index);
            segment_index++;
        }
    }

    // 5. Cleanup
    vad_iterator_free(&vad);
    wav_reader_close(&reader);

    return EXIT_SUCCESS;
}
