/*
    silero_vad.c - C23 Implementation of Silero VAD using ONNX Runtime C API
    Translated from silero-vad-onnx.cpp.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdckdint.h>
#include <stddef.h>

#include "silero_vad.h"
#include "wav.h"

/* --- Platform Specifics for ONNX Runtime --- */
typedef char ort_char_t;

// Helper to convert char* path to ONNX compatible path
[[nodiscard]]
static ort_char_t* create_ort_path(const char* path) {
    if (path == nullptr) {
        return nullptr;
    }

    auto dup = strdup(path);
    if (dup == nullptr) {
        return nullptr;
    }
    return dup;
}

static void free_ort_path(ort_char_t* path) {
    free(path);
}

static bool is_16k_model(const char* path) {
    if (path == nullptr) {
        return false;
    }
    return strstr(path, "16k") != nullptr;
}

/* --- Constants --- */
//#define DEBUG_SPEECH_PROB

static void vec_init(timestamp_vector_t* vec) {
    vec->data = nullptr;
    vec->size = 0;
    vec->capacity = 0;
}

static void vec_push(timestamp_vector_t* vec, timestamp_t ts) {
    if (vec->size >= vec->capacity) {
        constexpr size_t initial_capacity = 8;
        const auto new_cap = vec->capacity == 0 ? initial_capacity : vec->capacity * 2;
        auto new_data = (timestamp_t*)realloc(vec->data, new_cap * sizeof(timestamp_t));
        if (new_data == nullptr) {
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

// Check ONNX Status helper
static void check_status(const OrtApi* g_ort, OrtStatus* status) {
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "ONNX Runtime Error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(EXIT_FAILURE);
    }
}

void vad_iterator_reset_states(vad_iterator_t* vad) {
    if (vad == nullptr || vad->state == nullptr || vad->context == nullptr) {
        return;
    }

    memset(vad->state, 0, vad->size_state * sizeof(float));
    memset(vad->context, 0, vad->context_samples * sizeof(float));
    
    vad->triggered = false;
    vad->temp_end = 0U;
    vad->current_sample = 0U;
    vad->prev_end = 0;
    vad->next_start = 0;
    
    vec_clear(&vad->speeches);
    vad->current_speech = (timestamp_t){-1, -1};
}

[[nodiscard]]
bool vad_iterator_init(vad_iterator_t* vad, const char* model_path,
                       int sample_rate, int window_frame_size_ms,
                       float threshold, int min_silence_ms,
                       int speech_pad_ms, int min_speech_ms,
                       float max_speech_s)
{
    if (vad == nullptr || model_path == nullptr) {
        return false;
    }

    memset(vad, 0, sizeof(*vad));
    
    // 1. Setup API
    vad->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (vad->g_ort == nullptr) {
        fprintf(stderr, "Failed to init ONNX Runtime API\n");
        return false;
    }

    // 2. Constants & Sizes
    const bool model_is_16k_only = is_16k_model(model_path);
    if (model_is_16k_only && sample_rate != 16'000) {
        fprintf(stderr, "Model at %s supports only 16'000 Hz\n", model_path);
        return false;
    }
    if (!model_is_16k_only && sample_rate != 16'000 && sample_rate != 8'000) {
        fprintf(stderr, "Supported sample rates: 8'000 or 16'000 Hz (got %d)\n", sample_rate);
        return false;
    }

    const int context_samples = sample_rate == 16'000 ? 64 : 32;
    constexpr unsigned int state_channels = 2U;
    constexpr unsigned int state_batch = 1U;
    constexpr unsigned int state_width = 128U;

    vad->sample_rate = sample_rate;
    vad->sr_per_ms = sample_rate / 1'000;
    vad->window_size_samples = window_frame_size_ms * vad->sr_per_ms;
    vad->context_samples = context_samples;
    vad->effective_window_size = vad->window_size_samples + vad->context_samples;
    vad->size_state = state_channels * state_batch * state_width;
    
    vad->threshold = threshold;
    vad->min_silence_samples = vad->sr_per_ms * min_silence_ms;
    vad->speech_pad_samples = vad->sr_per_ms * speech_pad_ms;
    vad->min_speech_samples = vad->sr_per_ms * min_speech_ms;
    vad->max_speech_samples = (sample_rate * max_speech_s - vad->window_size_samples - 2 * vad->speech_pad_samples);
    vad->min_silence_samples_at_max_speech = vad->sr_per_ms * 98;

    // 3. Allocate Buffers
    vad->context = (float*)calloc((size_t)vad->context_samples, sizeof(float));
    vad->state = (float*)calloc((size_t)vad->size_state, sizeof(float));
    vad->sr_tensor_data = (int64_t*)calloc(1, sizeof(int64_t));
    vad->input_buffer = (float*)calloc((size_t)vad->effective_window_size, sizeof(float));
    vec_init(&vad->speeches);
    vad->current_speech = (timestamp_t){-1, -1};

    if (vad->context == nullptr || vad->state == nullptr || vad->sr_tensor_data == nullptr || vad->input_buffer == nullptr) {
        vad_iterator_free(vad);
        return false;
    }

    *vad->sr_tensor_data = sample_rate;

    // 4. Setup ONNX Env & Session
    check_status(vad->g_ort, vad->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SileroVAD", &vad->env));
    check_status(vad->g_ort, vad->g_ort->CreateSessionOptions(&vad->session_options));

    constexpr int ort_thread_count = 1;
    const auto g = vad->g_ort;
    const auto opts = vad->session_options;

    check_status(g, g->SetIntraOpNumThreads(opts, ort_thread_count));
    check_status(g, g->SetInterOpNumThreads(opts, ort_thread_count));
    check_status(g, g->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

    ort_char_t* ort_path = create_ort_path(model_path);
    if (ort_path == nullptr) {
        vad_iterator_free(vad);
        return false;
    }
    
    check_status(vad->g_ort, vad->g_ort->CreateSession(vad->env, ort_path, vad->session_options, &vad->session));
    free_ort_path(ort_path);

    check_status(vad->g_ort, vad->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &vad->memory_info));

    return true;
}

void vad_iterator_free(vad_iterator_t* vad) {
    if (vad == nullptr) {
        return;
    }

    if (vad->g_ort != nullptr) {
        if (vad->session != nullptr) vad->g_ort->ReleaseSession(vad->session);
        if (vad->session_options != nullptr) vad->g_ort->ReleaseSessionOptions(vad->session_options);
        if (vad->env != nullptr) vad->g_ort->ReleaseEnv(vad->env);
        if (vad->memory_info != nullptr) vad->g_ort->ReleaseMemoryInfo(vad->memory_info);
        vad->session = nullptr;
        vad->session_options = nullptr;
        vad->env = nullptr;
        vad->memory_info = nullptr;
        vad->g_ort = nullptr;
    }
    
    free(vad->context);
    free(vad->state);
    free(vad->sr_tensor_data);
    free(vad->input_buffer);
    vad->context = nullptr;
    vad->state = nullptr;
    vad->sr_tensor_data = nullptr;
    vad->input_buffer = nullptr;
    vec_free(&vad->speeches);
}

// Core inference logic
static void vad_predict(vad_iterator_t* vad, const float* data_chunk) {
    if (vad == nullptr || data_chunk == nullptr) {
        return;
    }

    const auto g = vad->g_ort;
    if (g == nullptr || vad->memory_info == nullptr || vad->session == nullptr) {
        return;
    }

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
    const auto speech_prob = output_data[0];

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
    if (vad == nullptr || input_wav == nullptr) {
        return;
    }

    if (vad->window_size_samples == 0) {
        return;
    }

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
