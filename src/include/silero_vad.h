#ifndef SILERO_VAD_H_
#define SILERO_VAD_H_

#include <onnxruntime_c_api.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  int start;
  int end;
} timestamp_t;

typedef struct {
  timestamp_t *data;
  size_t size;
  size_t capacity;
} timestamp_vector_t;

typedef struct {
  // ONNX Runtime Resources
  const OrtApi *g_ort;
  OrtEnv *env;
  OrtSession *session;
  OrtSessionOptions *session_options;
  OrtMemoryInfo *memory_info;
  OrtAllocator *allocator;

  // Buffers and State
  float *context;
  float *state;
  int64_t *sr_tensor_data;

  // Persistent Input/Output Buffers
  float *input_buffer;

  // Configuration
  int sample_rate;
  int sr_per_ms;
  int window_size_samples;
  int effective_window_size;
  int context_samples;
  unsigned int size_state;

  // Thresholds
  float threshold;
  int min_silence_samples;
  int min_silence_samples_at_max_speech;
  int min_speech_samples;
  float max_speech_samples;
  int speech_pad_samples;

  // Logic State
  bool triggered;
  unsigned int temp_end;
  unsigned int current_sample;
  int prev_end;
  int next_start;

  timestamp_t current_speech;
  timestamp_vector_t speeches;

} vad_iterator_t;

[[nodiscard]]
bool vad_iterator_init(vad_iterator_t *vad, const char *model_path,
                       int sample_rate, int window_frame_size_ms,
                       float threshold, int min_silence_ms, int speech_pad_ms,
                       int min_speech_ms, float max_speech_s);

void vad_iterator_reset_states(vad_iterator_t *vad);
void vad_iterator_process(vad_iterator_t *vad, const float *input_wav,
                          size_t audio_length_samples);
void vad_iterator_free(vad_iterator_t *vad);

#endif /* SILERO_VAD_H_ */
