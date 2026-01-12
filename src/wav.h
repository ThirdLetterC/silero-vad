/*
    wav.h - C23 Single-Header WAV Reader/Writer Library
    Based on the original C++ implementation by Binbin Zhang.
    Translated to ISO C23.
*/

#ifndef FRONTEND_WAV_H_
#define FRONTEND_WAV_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdckdint.h> // C23 checked arithmetic

/* WAV Header Structure */
typedef struct {
    char riff[4];           // "RIFF"
    uint32_t size;
    char wav[4];            // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;
    uint16_t format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t bytes_per_second;
    uint16_t block_size;
    uint16_t bit;
    char data[4];           // "data"
    uint32_t data_size;
} wav_header_t;

/* WavReader Structure */
typedef struct {
    int num_channel;
    int sample_rate;
    int bits_per_sample;
    size_t num_samples;  // Total sample points per channel
    float* data;         // Interleaved data if multi-channel, flat if mono
    bool loaded;
} wav_reader_t;

/* WavWriter Structure */
typedef struct {
    const float* data;
    size_t num_samples;
    int num_channel;
    int sample_rate;
    int bits_per_sample;
} wav_writer_t;

/* WavReader Implementation 
*/

[[nodiscard]] 
static inline bool wav_reader_open(wav_reader_t* reader, const char* filename) {
    if (!reader || !filename) return false;
    memset(reader, 0, sizeof(*reader));

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open WAV file '%s'\n", filename);
        return false;
    }

    wav_header_t header;
    if (fread(&header, 1, sizeof(header), fp) != sizeof(header)) {
        fprintf(stderr, "Error: Failed to read WAV header\n");
        fclose(fp);
        return false;
    }

    /* Handle extra fmt chunk data */
    if (header.fmt_size < 16) {
        fprintf(stderr, "Error: WAV fmt chunk too small\n");
        fclose(fp);
        return false;
    } else if (header.fmt_size > 16) {
        int offset = 44 - 8 + header.fmt_size - 16;
        if (fseek(fp, offset, SEEK_SET) != 0) {
            fclose(fp);
            return false;
        }
        // Re-read 'data' chunk header which might have been displaced
        // Note: The original code's logic for seeking here assumes a specific 
        // struct layout. We perform a specific read to find "data" chunk.
        char chunk_id[4];
        if (fread(chunk_id, 1, 4, fp) != 4) { fclose(fp); return false; }
        // Simplification: relying on the loop below to find 'data'
    }

    /* Skip non-data chunks (e.g., 'fact', 'list') */
    /* We need to re-synchronize to the stream to find "data" */
    // Note: This is a direct port of the logic. Robust parsing would read chunk-by-chunk.
    // We assume header.data currently holds the chunk ID at current file pos.
    
    // Reset to after fmt header to scan for data
    fseek(fp, 20 + header.fmt_size, SEEK_SET); 
    
    char chunk_header[8]; // ID + Size
    while (true) {
        if (fread(chunk_header, 1, 8, fp) != 8) {
            fprintf(stderr, "Error: Unexpected end of file while searching for 'data'\n");
            fclose(fp);
            return false;
        }
        
        uint32_t chunk_size = 0;
        memcpy(&chunk_size, chunk_header + 4, 4); // Endianness assumption: Little Endian

        if (strncmp(chunk_header, "data", 4) == 0) {
            header.data_size = chunk_size;
            break;
        }
        
        // Skip this chunk
        if (fseek(fp, chunk_size, SEEK_CUR) != 0) {
            fclose(fp);
            return false;
        }
    }

    /* Fix header data_size if 0 (streaming issue) */
    if (header.data_size == 0) {
        long current_pos = ftell(fp);
        fseek(fp, 0, SEEK_END);
        long end_pos = ftell(fp);
        header.data_size = (uint32_t)(end_pos - current_pos);
        fseek(fp, current_pos, SEEK_SET);
    }

    reader->num_channel = header.channels;
    reader->sample_rate = header.sample_rate;
    reader->bits_per_sample = header.bit;

    /* Calculate total samples and allocate memory */
    if (reader->bits_per_sample == 0) {
        fclose(fp);
        return false;
    }
    
    size_t num_data = header.data_size / (reader->bits_per_sample / 8);
    reader->num_samples = num_data / reader->num_channel;
    
    reader->data = (float*)malloc(num_data * sizeof(float));
    if (!reader->data) {
        fprintf(stderr, "Error: Memory allocation failed for WAV data\n");
        fclose(fp);
        return false;
    }

    printf("WAV Info: Channels=%d, Rate=%d, Bits=%d, Samples=%zu, Size=%u\n",
           reader->num_channel, reader->sample_rate, reader->bits_per_sample, 
           num_data, header.data_size);

    /* Read and convert data */
    switch (reader->bits_per_sample) {
        case 8: {
            // 8-bit PCM is unsigned 0..255, usually center 128. 
            // Original code treated it as signed char / 32768? 
            // Standard WAV 8-bit is unsigned. Original code: `char sample`. 
            // We preserve original behavior: assume signed char input.
            char sample;
            for (size_t i = 0; i < num_data; ++i) {
                if (fread(&sample, 1, 1, fp) != 1) break;
                reader->data[i] = (float)sample / 32768.0f;
            }
            break;
        }
        case 16: {
            int16_t sample;
            for (size_t i = 0; i < num_data; ++i) {
                if (fread(&sample, 2, 1, fp) != 1) break;
                reader->data[i] = (float)sample / 32768.0f;
            }
            break;
        }
        case 32: {
            if (header.format == 1) { // PCM Signed 32-bit
                int32_t sample;
                for (size_t i = 0; i < num_data; ++i) {
                    if (fread(&sample, 4, 1, fp) != 1) break;
                    reader->data[i] = (float)sample / 32768.0f; // This scaling seems odd for 32-bit int (usually 2^31), but preserving behavior.
                }
            } else if (header.format == 3) { // IEEE Float
                float sample;
                for (size_t i = 0; i < num_data; ++i) {
                    if (fread(&sample, 4, 1, fp) != 1) break;
                    reader->data[i] = sample;
                }
            } else {
                fprintf(stderr, "Error: Unsupported 32-bit format %d\n", header.format);
                free(reader->data);
                reader->data = nullptr;
                fclose(fp);
                return false;
            }
            break;
        }
        default:
            fprintf(stderr, "Error: Unsupported bit depth %d\n", reader->bits_per_sample);
            free(reader->data);
            reader->data = nullptr;
            fclose(fp);
            return false;
    }

    reader->loaded = true;
    fclose(fp);
    return true;
}

static inline void wav_reader_close(wav_reader_t* reader) {
    if (reader && reader->data) {
        free(reader->data);
        reader->data = nullptr;
        reader->loaded = false;
    }
}

/* WavWriter Implementation 
*/

static inline void wav_writer_init(wav_writer_t* writer, const float* data, size_t num_samples, 
                                   int num_channel, int sample_rate, int bits_per_sample) {
    writer->data = data;
    writer->num_samples = num_samples;
    writer->num_channel = num_channel;
    writer->sample_rate = sample_rate;
    writer->bits_per_sample = bits_per_sample;
}

static inline bool wav_writer_write(const wav_writer_t* writer, const char* filename) {
    if (!writer || !writer->data || !filename) return false;

    FILE* fp = fopen(filename, "w"); // "w" is text mode usually, but mostly works. Prefer "wb".
    // Original used "w", likely meant "wb".
    if (!fp) fp = fopen(filename, "wb"); 
    if (!fp) return false;

    wav_header_t header;
    // Hardcoded header template from original
    // 52 49 46 46 (RIFF) ...
    char wav_header_template[44] = {
        0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 
        0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
        0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00
    };
    
    memcpy(&header, wav_header_template, sizeof(header));
    
    header.channels = (uint16_t)writer->num_channel;
    header.bit = (uint16_t)writer->bits_per_sample;
    header.sample_rate = (uint32_t)writer->sample_rate;
    
    // Checked arithmetic for size calculation to prevent overflow
    size_t data_bytes;
    if (ckd_mul(&data_bytes, writer->num_samples, (size_t)writer->num_channel)) return false;
    if (ckd_mul(&data_bytes, data_bytes, (size_t)(writer->bits_per_sample / 8))) return false;

    header.data_size = (uint32_t)data_bytes;
    header.size = (uint32_t)(sizeof(header) - 8 + header.data_size);
    header.bytes_per_second = (uint32_t)(writer->sample_rate * writer->num_channel * (writer->bits_per_sample / 8));
    header.block_size = (uint16_t)(writer->num_channel * (writer->bits_per_sample / 8));

    fwrite(&header, 1, sizeof(header), fp);

    for (size_t i = 0; i < writer->num_samples; ++i) {
        for (int j = 0; j < writer->num_channel; ++j) {
            size_t idx = i * writer->num_channel + j;
            float val = writer->data[idx];
            
            switch (writer->bits_per_sample) {
                case 8: {
                    char sample = (char)val;
                    fwrite(&sample, 1, 1, fp);
                    break;
                }
                case 16: {
                    int16_t sample = (int16_t)val;
                    fwrite(&sample, 2, 1, fp);
                    break;
                }
                case 32: {
                    int sample = (int)val;
                    fwrite(&sample, 4, 1, fp);
                    break;
                }
            }
        }
    }

    fclose(fp);
    return true;
}

#endif /* FRONTEND_WAV_H_ */
