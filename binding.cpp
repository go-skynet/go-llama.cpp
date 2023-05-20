#include "common.h"
#include "llama.h"
#include "binding.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
            _exit(130);
    }
}
#endif

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_context* ctx = (llama_context*) state_pr;
    gpt_params params = *params_p;

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }
    
    std::mt19937 rng(params.seed);

    llama_init_backend();
  
    int n_past = 0;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (embd_inp.size() > 0) {
        if (llama_eval(ctx, embd_inp.data(), embd_inp.size(), n_past, params.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }
    }

    const int n_embd = llama_n_embd(ctx);

    const auto embeddings = llama_get_embeddings(ctx);

    for (int i = 0; i < n_embd; i++) {
        res_embeddings[i]=embeddings[i];
    }
        
    return 0;
}


int get_token_embeddings(void* params_ptr, void* state_pr,  int *tokens, int tokenSize, float * res_embeddings) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_context* ctx = (llama_context*) state_pr;
    gpt_params params = *params_p;
 
    for (int i = 0; i < tokenSize; i++) {
        auto token_str = llama_token_to_str(ctx, tokens[i]);
        if (token_str == nullptr) {
            continue;
        }
        std::vector<std::string> my_vector;
        std::string str_token(token_str); // create a new std::string from the char*
        params_p->prompt += str_token;
    }

  return get_embeddings(params_ptr,state_pr,res_embeddings);
}


int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_context* ctx = (llama_context*) state_pr;
  
    gpt_params params = *params_p;

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    std::mt19937 rng(params.seed);

    llama_init_backend();
  
    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    int n_past     = 0;
    int n_remain   = params.n_predict;
    int n_consumed = 0;

    std::vector<llama_token> embd;
    std::string res = "";

    while (n_remain != 0) {
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                n_past = std::max(1, params.n_keep);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 1;
                }
                n_past += n_eval;
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed) {
            // out of user input, sample next token
                const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[llama_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[llama_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;


            // call the token callback, no need to check if one is actually registered, that will
            // be handled on the Go side.
            auto token_str = llama_token_to_str(ctx, id);
            if (!tokenCallback(state_pr, (char*)token_str)) {
                break;
            }
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        for (auto id : embd) {
            res += llama_token_to_str(ctx, id);
        }

        // check for stop prompt
        if (params.antiprompt.size()) {
            std::string last_output;
            for (auto id : last_n_tokens) {
                last_output += llama_token_to_str(ctx, id);
            }
            // Check if each of the reverse prompts appears at the end of the output.
            for (std::string & antiprompt : params.antiprompt) {
                    //size_t extra_padding = params.interactive ? 0 : 2;
                    size_t extra_padding = 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos) {
                    goto end;
                }
            }
        }
      
        // end of text token
        if (embd.back() == llama_token_eos()) {
                break;
        }
    }

end:
#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    if (debug) {
        llama_print_timings(ctx);
        llama_reset_timings(ctx);
    }

    strcpy(result, res.c_str()); 
    return 0;
}

void llama_free_model(void *state_ptr) {
    llama_context* ctx = (llama_context*) state_ptr;
    llama_free(ctx);
}

void llama_free_params(void* params_ptr) {
    gpt_params* params = (gpt_params*) params_ptr;
    delete params;
}


std::vector<std::string> create_vector(const char** strings, int count) {
    std::vector<std::string>* vec = new std::vector<std::string>;
    for (int i = 0; i < count; i++) {
      vec->push_back(std::string(strings[i]));
    }
    return *vec;
}

void delete_vector(std::vector<std::string>* vec) {
    delete vec;
}

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, bool memory_f16, int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                             float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias ) {
    gpt_params* params = new gpt_params;
    params->seed = seed;
    params->n_threads = threads;
    params->n_predict = tokens;
    params->repeat_last_n = repeat_last_n;

    params->top_k = top_k;
    params->top_p = top_p;
    params->memory_f16 = memory_f16;
    params->temp = temp;
    params->repeat_penalty = repeat_penalty;
    params->n_batch = n_batch;
    params->n_keep = n_keep;

    if (ignore_eos) {
        params->logit_bias[llama_token_eos()] = -INFINITY;
    }
    if(antiprompt_count > 0) {
      params->antiprompt = create_vector(antiprompt, antiprompt_count);
    }
    params->tfs_z = tfs_z;
    params->typical_p = typical_p;
    params->presence_penalty = presence_penalty;
    params->mirostat = mirostat;
    params->mirostat_eta = mirostat_eta;
    params->mirostat_tau = mirostat_tau;
    params->penalize_nl = penalize_nl;
    std::stringstream ss(logit_bias);
    llama_token key;
    char sign;
    std::string value_str;
    if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
        params->logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
    } 
    params->frequency_penalty = frequency_penalty;
    params->prompt = prompt;
    
    return params;
}


void* load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, int n_gpu_layers) {
    // load the model
    auto lparams = llama_context_default_params();

    lparams.n_ctx      = n_ctx;
    lparams.seed       = n_seed;
    lparams.f16_kv     = memory_f16;
    lparams.embedding  = embeddings;
    lparams.use_mlock  = mlock;
    lparams.n_gpu_layers = n_gpu_layers;
    
    void* res = nullptr;
    try {
        res = llama_init_from_file(fname, lparams);
    } catch(std::runtime_error& e) {   
        fprintf(stderr, "failed %s",e.what());
        return res;
    }

    return res;
}
