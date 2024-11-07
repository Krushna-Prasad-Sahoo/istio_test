#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>  // Include time.h for measuring execution time

#define THRESHOLD 10.0f  // Define a threshold for noise detection
#define NUM_CHUNKS 50000 // Total 50 million records in chunks of 1000 records
#define CHUNK_SIZE 1000  // Number of records per chunk
#define NUM_COLS 50      // Number of columns per record

// Function prototypes
void *process_chunk(void *arg);
void detect_noise(float *data, int size);
void clean_data(float *data, int rows, int cols);
void report_progress(int thread_id, int chunk_id);

// Thread argument structure to pass chunk id and other information to thread
typedef struct {
    int chunk_id;
    int total_chunks;
    int num_threads;
} thread_arg_t;

int main(int argc, char **argv) {
    // Start the timer
    clock_t start_time = clock();  // Capture start time

    int num_threads = 4;  // For example, using 4 threads
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t *args = malloc(num_threads * sizeof(thread_arg_t));

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        args[i].chunk_id = i;
        args[i].total_chunks = NUM_CHUNKS;
        args[i].num_threads = num_threads;
        pthread_create(&threads[i], NULL, process_chunk, (void *)&args[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // End the timer
    clock_t end_time = clock();  // Capture end time

    // Calculate and print total execution time
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Total execution time: %f seconds\n", total_time);

    free(threads);
    free(args);
    return 0;
}

// Function to process a single chunk of data
void *process_chunk(void *arg) {
    thread_arg_t *args = (thread_arg_t *)arg;
    int chunk_id = args->chunk_id;
    int total_chunks = args->total_chunks;
    int num_threads = args->num_threads;

    // Dynamically allocate memory for a 1000 x 50 2D array (representing one chunk)
    float (*data)[NUM_COLS] = malloc(CHUNK_SIZE * NUM_COLS * sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize data with random values (representing the raw data)
    for (int i = 0; i < CHUNK_SIZE; i++) {
        for (int j = 0; j < NUM_COLS; j++) {
            data[i][j] = rand() % 20; // Random values for testing
        }
    }

    // Processing chunks distributed by thread id
    for (int i = chunk_id; i < total_chunks; i += num_threads) {
        if (i >= NUM_CHUNKS) {
            break; // Ensure we don't go out of bounds
        }
        detect_noise(data[i], NUM_COLS); // Detect noise on each record
        // After processing each chunk, report progress
        report_progress(chunk_id, i + 1); // i + 1 gives the 1-based chunk number
    }

    // Clean the noisy data after detection
    clean_data((float *)data, CHUNK_SIZE, NUM_COLS);

    // Free allocated memory
    free(data);
    return NULL;
}

// Function to detect noise in each record using a threshold
void detect_noise(float *data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] > THRESHOLD) {
            data[i] = -1; // Flag as noisy by setting to -1
        }
    }
}

// Function to clean flagged noisy data (replace -1 values with average of neighbors)
void clean_data(float *data, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (data[i * cols + j] == -1) { // Check if data is flagged as noisy
                // Simple cleaning: replace -1 with a default value or average of neighbors
                float replacement = (j > 0 ? data[i * cols + j - 1] : 0) +
                                    (j < cols - 1 ? data[i * cols + j + 1] : 0);
                data[i * cols + j] = replacement / 2; // Average of neighbors
            }
        }
    }
}

// Function to report progress (simple console output for each thread)
void report_progress(int thread_id, int chunk_id) {
    printf("Thread %d - Processed chunk %d\n", thread_id, chunk_id);
}
