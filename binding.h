#ifdef __cplusplus
#include <vector>
#include <string>
extern "C" {
#endif

#include <stdbool.h>

extern unsigned char tokenCallback(void *, char *);

void* load_model(const char *fname, int n_ctx, int n_parts, int n_seed, bool memory_f16, bool mlock);

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty, 
                            int repeat_last_n, bool ignore_eos, bool memory_f16, 
                            int n_batch, int n_keep, const char** antiprompt, int antiprompt_count);

void llama_free_params(void* params_ptr);

void llama_free_model(void* state);

int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug);

#ifdef __cplusplus
}


std::vector<std::string> create_vector(const char** strings, int count);
void delete_vector(std::vector<std::string>* vec);
#endif
