#ifndef NPY_LOADER_H
#define NPY_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Simple numpy .npy file loader for float32 arrays
// Supports only simple C-contiguous float32 arrays

bool load_npy_float32(const char* filename, float** data_out, int* rows_out, int* cols_out) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return false;
    }

    // Read magic string
    char magic[6];
    if (fread(magic, 1, 6, fp) != 6) {
        fprintf(stderr, "Error: Failed to read magic string\n");
        fclose(fp);
        return false;
    }

    // Check magic string
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        fprintf(stderr, "Error: Invalid numpy file format\n");
        fclose(fp);
        return false;
    }

    // Read version
    uint8_t major_version, minor_version;
    if (fread(&major_version, 1, 1, fp) != 1 ||
        fread(&minor_version, 1, 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to read version\n");
        fclose(fp);
        return false;
    }

    // Read header length
    uint16_t header_len = 0;
    if (major_version == 1) {
        if (fread(&header_len, 2, 1, fp) != 1) {
            fprintf(stderr, "Error: Failed to read header length\n");
            fclose(fp);
            return false;
        }
    } else if (major_version == 2 || major_version == 3) {
        uint32_t header_len_32;
        if (fread(&header_len_32, 4, 1, fp) != 1) {
            fprintf(stderr, "Error: Failed to read header length\n");
            fclose(fp);
            return false;
        }
        header_len = (uint16_t)header_len_32;
    } else {
        fprintf(stderr, "Error: Unsupported numpy version %d.%d\n", major_version, minor_version);
        fclose(fp);
        return false;
    }

    // Read header
    char* header = (char*)malloc(header_len + 1);
    if (fread(header, 1, header_len, fp) != header_len) {
        fprintf(stderr, "Error: Failed to read header\n");
        free(header);
        fclose(fp);
        return false;
    }
    header[header_len] = '\0';

    // Parse shape from header
    // Header format: "{'descr': '<f4', 'fortran_order': False, 'shape': (rows, cols), }"
    // or for 1D: "{'descr': '<f4', 'fortran_order': False, 'shape': (size,), }"
    
    char* shape_start = strstr(header, "'shape': (");
    if (!shape_start) {
        shape_start = strstr(header, "\"shape\": (");
    }
    
    if (!shape_start) {
        fprintf(stderr, "Error: Cannot find shape in header\n");
        free(header);
        fclose(fp);
        return false;
    }

    shape_start += 10; // Move past "'shape': ("
    
    int rows = 0, cols = 0;
    int dims_parsed = sscanf(shape_start, "%d, %d)", &rows, &cols);
    
    if (dims_parsed == 1) {
        // 1D array
        cols = 1;
    } else if (dims_parsed != 2) {
        fprintf(stderr, "Error: Cannot parse shape from header\n");
        free(header);
        fclose(fp);
        return false;
    }

    free(header);

    // Calculate total size
    size_t total_elements = (size_t)rows * cols;
    
    // Allocate memory
    float* data = (float*)malloc(total_elements * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate memory for data\n");
        fclose(fp);
        return false;
    }

    // Read data
    size_t elements_read = fread(data, sizeof(float), total_elements, fp);
    if (elements_read != total_elements) {
        fprintf(stderr, "Error: Failed to read data (expected %zu, got %zu)\n", 
                total_elements, elements_read);
        free(data);
        fclose(fp);
        return false;
    }

    fclose(fp);

    *data_out = data;
    *rows_out = rows;
    *cols_out = cols;

    return true;
}

#endif // NPY_LOADER_H
