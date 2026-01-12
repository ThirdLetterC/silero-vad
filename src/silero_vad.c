/*
    silero_vad.c - C23 Implementation of Silero VAD using ONNX Runtime C API
    Translated from silero-vad-onnx.cpp.
*/

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdckdint.h>

#include "wav.h"
#include <onnxruntime_c_api.h>

/* --- Platform Specifics for ONNX Runtime --- */
#ifdef _WIN32
    #include <Windows.h>
    typedef wchar_t ort_char_t;
#else
    typedef char ort_char_t;
#endif

// Helper to convert char* path to ONNX compatible path
static ort_char_t* create_ort_path(const char* path) {
#ifdef _WIN32
    int len = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (len <= 0) return nullptr;
    wchar_t* wpath = (wchar_t*)malloc(len * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, len);
    return wpath;
#else
    return strdup(path);
#endif
}

static void free_ort_path(ort_char_t* path) {
    free(path);
}

/* --- Constants --- */
//#define DEBUG_SPEECH_PROB

/* --- Data Structures --- */

typedef struct {
    int start;
    int end;
} timestamp_t;

// Dynamic array for timestamps (replacing std::vector<timestamp_t>)
typedef struct {
    timestamp_t* data;
    size_t size;
    size_t capacity;
} timestamp_vector_t;

static void vec_init(timestamp_vector_t* vec) {
    vec->data = nullptr;
    vec->size = 0;
    vec->capacity = 0;
}

static void vec_push(timestamp_vector_t* vec, timestamp_t ts) {
    if (vec->size >= vec->capacity) {
        size_t new_cap = vec->capacity == 0 ? 8 : vec->capacity * 2;
        timestamp_t* new_data = (timestamp_t*)realloc(vec->data, new_cap * sizeof(timestamp_t));
        if (!new_data) {
            fprintf(stderr, "OOM in vec_push\n");
            exit(EXIT_FAILURE);
        }
        vec->data = new_data;
        vec->capacity = new_cap;
    }
    vec->data[vec->size++] = ts;
}

static void vec_free(timestamp_vector_t* vec) {
    free(vec->data);
    vec_init(vec);
}

static void vec_clear(timestamp_vector_t* vec) {
    vec->size = 0;
}

/* --- VadIterator --- */

typedef struct {
    // ONNX Runtime Resources
    const OrtApi* g_ort;
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* session_options;
    OrtMemoryInfo* memory_info;
    OrtAllocator* allocator; // Default allocator

    // Buffers and State
    float* context;             // Holds 64 samples
    float* state;               // Holds state tensor data (2 * 1 * 128)
    int64_t* sr_tensor_data;    // Holds sample rate
    
    // Persistent Input/Output Buffers to avoid reallocation
    float* input_buffer;        // effective_window_size
    
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

/* --- VadIterator Implementation --- */

// Check ONNX Status helper
static void check_status(const OrtApi* g_ort, OrtStatus* status) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "ONNX Runtime Error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(EXIT_FAILURE);
    }
}

void vad_iterator_reset_states(vad_iterator_t* vad) {
    memset(vad->state, 0, vad->size_state * sizeof(float));
    memset(vad->context, 0, vad->context_samples * sizeof(float));
    
    vad->triggered = false;
    vad->temp_end = 0;
    vad->current_sample = 0;
    vad->prev_end = 0;
    vad->next_start = 0;
    
    vec_clear(&vad->speeches);
    vad->current_speech = (timestamp_t){-1, -1};
}

bool vad_iterator_init(vad_iterator_t* vad, const char* model_path, 
                       int sample_rate, int window_frame_size_ms, 
                       float threshold, int min_silence_ms, 
                       int speech_pad_ms, int min_speech_ms, 
                       float max_speech_s) 
{
    memset(vad, 0, sizeof(*vad));
    
    // 1. Setup API
    vad->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!vad->g_ort) {
        fprintf(stderr, "Failed to init ONNX Runtime API\n");
        return false;
    }

    // 2. Constants & Sizes
    vad->sample_rate = sample_rate;
    vad->sr_per_ms = sample_rate / 1000;
    vad->window_size_samples = window_frame_size_ms * vad->sr_per_ms;
    vad->context_samples = 64;
    vad->effective_window_size = vad->window_size_samples + vad->context_samples;
    vad->size_state = 2 * 1 * 128;
    
    vad->threshold = threshold;
    vad->min_silence_samples = vad->sr_per_ms * min_silence_ms;
    vad->speech_pad_samples = vad->sr_per_ms * speech_pad_ms;
    vad->min_speech_samples = vad->sr_per_ms * min_speech_ms;
    vad->max_speech_samples = (sample_rate * max_speech_s - vad->window_size_samples - 2 * vad->speech_pad_samples);
    vad->min_silence_samples_at_max_speech = vad->sr_per_ms * 98;

    // 3. Allocate Buffers
    vad->context = (float*)calloc(vad->context_samples, sizeof(float));
    vad->state = (float*)calloc(vad->size_state, sizeof(float));
    vad->sr_tensor_data = (int64_t*)malloc(sizeof(int64_t));
    *vad->sr_tensor_data = sample_rate;
    vad->input_buffer = (float*)malloc(vad->effective_window_size * sizeof(float));
    
    vec_init(&vad->speeches);
    vad->current_speech = (timestamp_t){-1, -1};

    // 4. Setup ONNX Env & Session
    check_status(vad->g_ort, vad->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SileroVAD", &vad->env));
    check_status(vad->g_ort, vad->g_ort->CreateSessionOptions(&vad->session_options));
    
    vad->g_ort->SetIntraOpNumThreads(vad->session_options, 1);
    vad->g_ort->SetInterOpNumThreads(vad->session_options, 1);
    vad->g_ort->SetSessionGraphOptimizationLevel(vad->session_options, ORT_ENABLE_ALL);

    ort_char_t* ort_path = create_ort_path(model_path);
    if (!ort_path) return false;
    
    check_status(vad->g_ort, vad->g_ort->CreateSession(vad->env, ort_path, vad->session_options, &vad->session));
    free_ort_path(ort_path);

    check_status(vad->g_ort, vad->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &vad->memory_info));

    return true;
}

void vad_iterator_free(vad_iterator_t* vad) {
    if (!vad->g_ort) return;
    
    if (vad->session) vad->g_ort->ReleaseSession(vad->session);
    if (vad->session_options) vad->g_ort->ReleaseSessionOptions(vad->session_options);
    if (vad->env) vad->g_ort->ReleaseEnv(vad->env);
    if (vad->memory_info) vad->g_ort->ReleaseMemoryInfo(vad->memory_info);

    free(vad->context);
    free(vad->state);
    free(vad->sr_tensor_data);
    free(vad->input_buffer);
    vec_free(&vad->speeches);
}

// Core inference logic
static void vad_predict(vad_iterator_t* vad, const float* data_chunk) {
    const OrtApi* g = vad->g_ort;

    // 1. Prepare Input Buffer: [Context (64)] + [Chunk (WindowSize)]
    memcpy(vad->input_buffer, vad->context, vad->context_samples * sizeof(float));
    memcpy(vad->input_buffer + vad->context_samples, data_chunk, vad->window_size_samples * sizeof(float));

    // 2. Create Tensors
    OrtValue* input_ort = nullptr;
    int64_t input_dims[] = {1, vad->effective_window_size};
    check_status(g, g->CreateTensorWithDataAsOrtValue(vad->memory_info, vad->input_buffer, 
        vad->effective_window_size * sizeof(float), input_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_ort));

    OrtValue* state_ort = nullptr;
    int64_t state_dims[] = {2, 1, 128};
    check_status(g, g->CreateTensorWithDataAsOrtValue(vad->memory_info, vad->state, 
        vad->size_state * sizeof(float), state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_ort));

    OrtValue* sr_ort = nullptr;
    int64_t sr_dims[] = {1};
    check_status(g, g->CreateTensorWithDataAsOrtValue(vad->memory_info, vad->sr_tensor_data, 
        sizeof(int64_t), sr_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_ort));

    // 3. Run Inference
    const char* input_names[] = {"input", "state", "sr"};
    const char* output_names[] = {"output", "stateN"};
    const OrtValue* inputs[] = {input_ort, state_ort, sr_ort};
    OrtValue* outputs[] = {nullptr, nullptr};

    check_status(g, g->Run(vad->session, nullptr, input_names, inputs, 3, output_names, 2, outputs));

    // 4. Get Outputs
    float* output_data = nullptr;
    check_status(g, g->GetTensorMutableData(outputs[0], (void**)&output_data));
    float speech_prob = output_data[0];

    float* stateN_data = nullptr;
    check_status(g, g->GetTensorMutableData(outputs[1], (void**)&stateN_data));
    
    // Update state for next step
    memcpy(vad->state, stateN_data, vad->size_state * sizeof(float));
    
    // Cleanup Tensors (wrappers only, data is owned by struct)
    g->ReleaseValue(input_ort);
    g->ReleaseValue(state_ort);
    g->ReleaseValue(sr_ort);
    g->ReleaseValue(outputs[0]);
    g->ReleaseValue(outputs[1]);

    // 5. Logic
    vad->current_sample += (unsigned int)vad->window_size_samples;

    if (speech_prob >= vad->threshold) {
#ifdef DEBUG_SPEECH_PROB
        float speech = (float)vad->current_sample - vad->window_size_samples;
        printf("{ start: %.3f s (%.3f) %08d}\n", speech / vad->sample_rate, speech_prob, vad->current_sample - vad->window_size_samples);
#endif
        if (vad->temp_end != 0) {
            vad->temp_end = 0;
            if (vad->next_start < vad->prev_end)
                vad->next_start = vad->current_sample - vad->window_size_samples;
        }
        if (!vad->triggered) {
            vad->triggered = true;
            vad->current_speech.start = vad->current_sample - vad->window_size_samples;
        }
        
        // Update context: Last 64 samples of current buffer
        // In C++: copy(new_data.end() - context_samples, ...)
        memcpy(vad->context, vad->input_buffer + (vad->effective_window_size - vad->context_samples), vad->context_samples * sizeof(float));
        return;
    }

    if (vad->triggered && ((int)(vad->current_sample - vad->current_speech.start) > vad->max_speech_samples)) {
        if (vad->prev_end > 0) {
            vad->current_speech.end = vad->prev_end;
            vec_push(&vad->speeches, vad->current_speech);
            
            vad->current_speech = (timestamp_t){-1, -1};
            if (vad->next_start < vad->prev_end)
                vad->triggered = false;
            else
                vad->current_speech.start = vad->next_start;
            
            vad->prev_end = 0;
            vad->next_start = 0;
            vad->temp_end = 0;
        } else {
            vad->current_speech.end = vad->current_sample;
            vec_push(&vad->speeches, vad->current_speech);
            vad->current_speech = (timestamp_t){-1, -1};
            vad->prev_end = 0;
            vad->next_start = 0;
            vad->temp_end = 0;
            vad->triggered = false;
        }
        memcpy(vad->context, vad->input_buffer + (vad->effective_window_size - vad->context_samples), vad->context_samples * sizeof(float));
        return;
    }

    if ((speech_prob >= (vad->threshold - 0.15f)) && (speech_prob < vad->threshold)) {
        memcpy(vad->context, vad->input_buffer + (vad->effective_window_size - vad->context_samples), vad->context_samples * sizeof(float));
        return;
    }

    if (speech_prob < (vad->threshold - 0.15f)) {
#ifdef DEBUG_SPEECH_PROB
        float speech = (float)vad->current_sample - vad->window_size_samples - vad->speech_pad_samples;
        printf("{ end: %.3f s (%.3f) %08d}\n", speech / vad->sample_rate, speech_prob, vad->current_sample - vad->window_size_samples);
#endif
        if (vad->triggered) {
            if (vad->temp_end == 0)
                vad->temp_end = vad->current_sample;
            
            if ((int)(vad->current_sample - vad->temp_end) > vad->min_silence_samples_at_max_speech)
                vad->prev_end = vad->temp_end;
            
            if ((int)(vad->current_sample - vad->temp_end) >= vad->min_silence_samples) {
                vad->current_speech.end = vad->temp_end;
                if ((vad->current_speech.end - vad->current_speech.start) > vad->min_speech_samples) {
                    vec_push(&vad->speeches, vad->current_speech);
                    vad->current_speech = (timestamp_t){-1, -1};
                    vad->prev_end = 0;
                    vad->next_start = 0;
                    vad->temp_end = 0;
                    vad->triggered = false;
                }
            }
        }
        memcpy(vad->context, vad->input_buffer + (vad->effective_window_size - vad->context_samples), vad->context_samples * sizeof(float));
        return;
    }
}

void vad_iterator_process(vad_iterator_t* vad, const float* input_wav, size_t audio_length_samples) {
    vad_iterator_reset_states(vad);
    
    for (size_t j = 0; j < audio_length_samples; j += vad->window_size_samples) {
        if (j + vad->window_size_samples > audio_length_samples) break;
        vad_predict(vad, &input_wav[j]);
    }
    
    if (vad->current_speech.start >= 0) {
        vad->current_speech.end = (int)audio_length_samples;
        vec_push(&vad->speeches, vad->current_speech);
        vad->current_speech = (timestamp_t){-1, -1};
        vad->prev_end = 0;
        vad->next_start = 0;
        vad->temp_end = 0;
        vad->triggered = false;
    }
}

/* --- Main --- */

int main(void) {
    // 1. Read WAV
    const char* input_file = "audio/recorder.wav";
    wav_reader_t reader;
    
    printf("Loading WAV file: %s\n", input_file);
    if (!wav_reader_open(&reader, input_file)) {
        return EXIT_FAILURE;
    }

    // 2. Init VAD
    // Note: C23 auto/constexpr usage not critical here, keeping plain C types for clarity
    const char* model_path = "model/silero_vad.onnx";
    vad_iterator_t vad;
    
    // Default params matching C++ constructor defaults
    printf("Initializing VAD with model: %s\n", model_path);
    if (!vad_iterator_init(&vad, model_path, 16000, 32, 0.5f, 100, 30, 250, INFINITY)) {
        fprintf(stderr, "Failed to initialize VAD\n");
        wav_reader_close(&reader);
        return EXIT_FAILURE;
    }

    // 3. Process
    printf("Processing %zu samples...\n", reader.num_samples);
    vad_iterator_process(&vad, reader.data, reader.num_samples);

    // 4. Output Results
    const float sample_rate_float = 16000.0f;
    for (size_t i = 0; i < vad.speeches.size; i++) {
        timestamp_t ts = vad.speeches.data[i];
        float start_sec = rintf((ts.start / sample_rate_float) * 10.0f) / 10.0f;
        float end_sec = rintf((ts.end / sample_rate_float) * 10.0f) / 10.0f;
        
        printf("Speech detected from %.1f s to %.1f s\n", start_sec, end_sec);
    }

    // 5. Cleanup
    vad_iterator_free(&vad);
    wav_reader_close(&reader);

    return EXIT_SUCCESS;
}
