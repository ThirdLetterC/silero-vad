/*
    wav.h - C23 Single-Header WAV Reader/Writer Library
    Based on the original C++ implementation by Binbin Zhang.
    Translated to ISO C23.
*/

#ifndef FRONTEND_WAV_H_
#define FRONTEND_WAV_H_

#include <assert.h>
#include <stdckdint.h> // C23 checked arithmetic
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* WAV Header Structure */
typedef struct {
  char riff[4]; // "RIFF"
  uint32_t size;
  char wav[4]; // "WAVE"
  char fmt[4]; // "fmt "
  uint32_t fmt_size;
  uint16_t format;
  uint16_t channels;
  uint32_t sample_rate;
  uint32_t bytes_per_second;
  uint16_t block_size;
  uint16_t bit;
  char data[4]; // "data"
  uint32_t data_size;
} wav_header_t;
static_assert(sizeof(wav_header_t) == 44,
              "wav_header_t must match 44-byte WAV header");

/* WavReader Structure */
typedef struct {
  int num_channel;
  int sample_rate;
  int bits_per_sample;
  size_t num_samples; // Total sample points per channel
  float *data;        // Interleaved data if multi-channel, flat if mono
  bool loaded;
} wav_reader_t;

/* WavWriter Structure */
typedef struct {
  const float *data;
  size_t num_samples;
  int num_channel;
  int sample_rate;
  int bits_per_sample;
} wav_writer_t;

[[nodiscard]] bool wav_reader_open(wav_reader_t *reader, const char *filename);
void wav_reader_close(wav_reader_t *reader);

void wav_writer_init(wav_writer_t *writer, const float *data,
                     size_t num_samples, int num_channel, int sample_rate,
                     int bits_per_sample);
[[nodiscard]] bool wav_writer_write(const wav_writer_t *writer,
                                    const char *filename);

#endif /* FRONTEND_WAV_H_ */
