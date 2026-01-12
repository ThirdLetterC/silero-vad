#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wav.h"

static float clamp_sample(float v) {
  if (v > 1.0f) {
    return 1.0f;
  }
  if (v < -1.0f) {
    return -1.0f;
  }
  return v;
}

[[nodiscard]]
bool wav_reader_open(wav_reader_t *reader, const char *filename) {
  if (reader == nullptr || filename == nullptr)
    return false;
  memset(reader, 0, sizeof(*reader));

  FILE *fp = fopen(filename, "rb");
  if (fp == nullptr) {
    fprintf(stderr, "Error: Cannot open WAV file '%s'\n", filename);
    return false;
  }

  bool success = false;

  wav_header_t header;
  if (fread(&header, 1, sizeof(header), fp) != sizeof(header)) {
    fprintf(stderr, "Error: Failed to read WAV header\n");
    goto cleanup;
  }

  if (header.fmt_size < 16) {
    fprintf(stderr, "Error: WAV fmt chunk too small\n");
    goto cleanup;
  } else if (header.fmt_size > 16) {
    const long offset = 44L - 8L + (long)header.fmt_size - 16L;
    if (fseek(fp, offset, SEEK_SET) != 0) {
      goto cleanup;
    }
    constexpr size_t chunk_id_size = 4;
    char chunk_id[chunk_id_size];
    if (fread(chunk_id, 1, chunk_id_size, fp) != chunk_id_size) {
      goto cleanup;
    }
  }

  const long data_seek_offset = 20L + (long)header.fmt_size;
  if (fseek(fp, data_seek_offset, SEEK_SET) != 0) {
    goto cleanup;
  }

  constexpr size_t chunk_header_size = 8;
  char chunk_header[chunk_header_size];
  while (true) {
    if (fread(chunk_header, 1, chunk_header_size, fp) != chunk_header_size) {
      fprintf(stderr,
              "Error: Unexpected end of file while searching for 'data'\n");
      goto cleanup;
    }

    uint32_t chunk_size = 0;
    memcpy(&chunk_size, chunk_header + 4, sizeof(chunk_size));

    if (strncmp(chunk_header, "data", 4) == 0) {
      header.data_size = chunk_size;
      break;
    }

    if (fseek(fp, (long)chunk_size, SEEK_CUR) != 0) {
      goto cleanup;
    }
  }

  if (header.data_size == 0) {
    long current_pos = ftell(fp);
    if (current_pos < 0) {
      goto cleanup;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
      goto cleanup;
    }
    long end_pos = ftell(fp);
    if (end_pos < 0) {
      goto cleanup;
    }
    header.data_size = (uint32_t)(end_pos - current_pos);
    if (fseek(fp, current_pos, SEEK_SET) != 0) {
      goto cleanup;
    }
  }

  reader->num_channel = header.channels;
  reader->sample_rate = header.sample_rate;
  reader->bits_per_sample = header.bit;

  if (reader->num_channel == 0) {
    goto cleanup;
  }
  if (reader->bits_per_sample == 0) {
    goto cleanup;
  }

  size_t num_data = header.data_size / (reader->bits_per_sample / 8);
  reader->num_samples = num_data / reader->num_channel;

  reader->data = (float *)calloc(num_data, sizeof(float));
  if (reader->data == nullptr) {
    fprintf(stderr, "Error: Memory allocation failed for WAV data\n");
    goto cleanup;
  }

  printf("WAV Info: Channels=%d, Rate=%d, Bits=%d, Samples=%zu, Size=%u\n",
         reader->num_channel, reader->sample_rate, reader->bits_per_sample,
         num_data, header.data_size);

  size_t samples_read = 0;

  switch (reader->bits_per_sample) {
  case 8: {
    uint8_t sample = 0U;
    constexpr float inv_scale =
        1.0f / 127.5f; // Undo unsigned PCM offset and scale
    for (size_t i = 0; i < num_data; ++i) {
      if (fread(&sample, 1, 1, fp) != 1) {
        break;
      }
      reader->data[i] = (float)sample * inv_scale - 1.0f;
      samples_read++;
    }
    break;
  }
  case 16: {
    int16_t sample = 0;
    constexpr float inv_scale = 1.0f / 32'768.0f;
    for (size_t i = 0; i < num_data; ++i) {
      if (fread(&sample, 2, 1, fp) != 1) {
        break;
      }
      reader->data[i] = (float)sample * inv_scale;
      samples_read++;
    }
    break;
  }
  case 32: {
    if (header.format == 1) {
      int32_t sample = 0;
      constexpr float inv_scale = 1.0f / 2'147'483'648.0f;
      for (size_t i = 0; i < num_data; ++i) {
        if (fread(&sample, 4, 1, fp) != 1) {
          break;
        }
        reader->data[i] = (float)sample * inv_scale;
        samples_read++;
      }
    } else if (header.format == 3) {
      float sample = 0.0f;
      for (size_t i = 0; i < num_data; ++i) {
        if (fread(&sample, 4, 1, fp) != 1) {
          break;
        }
        reader->data[i] = sample;
        samples_read++;
      }
    } else {
      fprintf(stderr, "Error: Unsupported 32-bit format %d\n", header.format);
      goto cleanup;
    }
    break;
  }
  default:
    fprintf(stderr, "Error: Unsupported bit depth %d\n",
            reader->bits_per_sample);
    goto cleanup;
  }

  if (samples_read != num_data) {
    fprintf(stderr,
            "Error: WAV data truncated (expected %zu samples, got %zu)\n",
            num_data, samples_read);
    goto cleanup;
  }

  reader->loaded = true;
  success = true;

cleanup:
  fclose(fp);
  if (!success) {
    free(reader->data);
    reader->data = nullptr;
  }
  return success;
}

void wav_reader_close(wav_reader_t *reader) {
  if (reader != nullptr && reader->data != nullptr) {
    free(reader->data);
    reader->data = nullptr;
    reader->loaded = false;
  }
}

void wav_writer_init(wav_writer_t *writer, const float *data,
                     size_t num_samples, int num_channel, int sample_rate,
                     int bits_per_sample) {
  writer->data = data;
  writer->num_samples = num_samples;
  writer->num_channel = num_channel;
  writer->sample_rate = sample_rate;
  writer->bits_per_sample = bits_per_sample;
}

[[nodiscard]]
bool wav_writer_write(const wav_writer_t *writer, const char *filename) {
  if (writer == nullptr || writer->data == nullptr || filename == nullptr)
    return false;
  if (writer->num_channel <= 0 || writer->bits_per_sample <= 0 ||
      writer->sample_rate <= 0)
    return false;
  if ((writer->bits_per_sample % 8) != 0)
    return false;

  FILE *fp = fopen(filename, "wb");
  if (fp == nullptr)
    return false;

  bool success = false;

  wav_header_t header;
  constexpr char wav_header_template[] = {
      0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56,
      0x45, 0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00};

  memcpy(&header, wav_header_template, sizeof(header));

  header.channels = (uint16_t)writer->num_channel;
  header.bit = (uint16_t)writer->bits_per_sample;
  header.sample_rate = (uint32_t)writer->sample_rate;

  size_t data_bytes;
  if (ckd_mul(&data_bytes, writer->num_samples, (size_t)writer->num_channel))
    goto cleanup;
  if (ckd_mul(&data_bytes, data_bytes, (size_t)(writer->bits_per_sample / 8)))
    goto cleanup;
  if (data_bytes > UINT32_MAX)
    goto cleanup;

  header.data_size = (uint32_t)data_bytes;

  size_t riff_size = sizeof(header) - 8U;
  if (ckd_add(&riff_size, riff_size, data_bytes))
    goto cleanup;
  if (riff_size > UINT32_MAX)
    goto cleanup;

  size_t bytes_per_second = 0;
  if (ckd_mul(&bytes_per_second, (size_t)writer->sample_rate,
              (size_t)writer->num_channel))
    goto cleanup;
  if (ckd_mul(&bytes_per_second, bytes_per_second,
              (size_t)(writer->bits_per_sample / 8)))
    goto cleanup;
  if (bytes_per_second > UINT32_MAX)
    goto cleanup;

  size_t block_size = 0;
  if (ckd_mul(&block_size, (size_t)writer->num_channel,
              (size_t)(writer->bits_per_sample / 8)))
    goto cleanup;
  if (block_size > UINT16_MAX)
    goto cleanup;

  header.size = (uint32_t)riff_size;
  header.bytes_per_second = (uint32_t)bytes_per_second;
  header.block_size = (uint16_t)block_size;

  if (fwrite(&header, 1, sizeof(header), fp) != sizeof(header))
    goto cleanup;

  for (size_t i = 0; i < writer->num_samples; ++i) {
    for (int j = 0; j < writer->num_channel; ++j) {
      size_t idx = i * writer->num_channel + j;
      const auto val = clamp_sample(writer->data[idx]);

      switch (writer->bits_per_sample) {
      case 8: {
        const auto sample = (uint8_t)lrintf((val + 1.0f) * 127.5f);
        if (fwrite(&sample, 1, 1, fp) != 1)
          goto cleanup;
        break;
      }
      case 16: {
        const auto sample = (int16_t)lrintf(val * 32'767.0f);
        if (fwrite(&sample, 2, 1, fp) != 1)
          goto cleanup;
        break;
      }
      case 32: {
        const auto sample = (int32_t)llrintf((double)val * 2'147'483'647.0);
        if (fwrite(&sample, 4, 1, fp) != 1)
          goto cleanup;
        break;
      }
      default:
        fprintf(stderr, "Error: Unsupported bit depth %d\n",
                writer->bits_per_sample);
        goto cleanup;
      }
    }
  }

  success = true;

cleanup:
  fclose(fp);
  return success;
}
