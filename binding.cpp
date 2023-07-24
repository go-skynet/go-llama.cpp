#include "common.h"
#include "llama.cpp"

#include "binding.h"
#include "grammar-parser.h"
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
#include <regex>
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
    llama_state* state = (llama_state*) state_pr;
    llama_context* ctx = state->ctx;
    gpt_params params = *params_p;

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }
    
    std::mt19937 rng(params.seed);
  
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
    llama_state* state = (llama_state*) state_pr;
    llama_context* ctx = state->ctx;
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

int eval(void* params_ptr,void* state_pr,char *text) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_state* state = (llama_state*) state_pr;
    llama_context* ctx = state->ctx;

    auto n_past = 0;
    auto last_n_tokens_data = std::vector<llama_token>(params_p->repeat_last_n, 0);

    auto tokens = std::vector<llama_token>(params_p->n_ctx);
    auto n_prompt_tokens = llama_tokenize(ctx, text, tokens.data(), tokens.size(), true);

    if (n_prompt_tokens < 1) {
        fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
        return 1;
    }

    // evaluate prompt
    return llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past, params_p->n_threads);
}
static llama_context ** g_ctx;

int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_state* state = (llama_state*) state_pr;
    llama_context* ctx = state->ctx;

    gpt_params params = *params_p;

    const int n_ctx = llama_n_ctx(ctx);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    std::mt19937 rng(params.seed);

    if (params.rope_freq_base != 10000.0) {
        fprintf(stderr, "%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        fprintf(stderr, "%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    if (params.n_ctx > 2048) {
        // TODO: determine the actual max context of the model (e.g. 4096 for LLaMA v2) and use that instead of 2048
        fprintf(stderr, "%s: warning: base model only supports context sizes no greater than 2048 tokens (%d specified)\n", __func__, params.n_ctx);
    } else if (params.n_ctx < 8) {
        fprintf(stderr, "%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }
    llama_context * ctx_guidance = NULL;
    g_ctx = &ctx;
    
    if (params.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        ctx_guidance = llama_new_context_with_model(state->model, lparams);
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        if (debug) {
            fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        }
        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            llama_set_rng_seed(ctx, params.seed);
            if (debug) {
                fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
            }
        } else {
            if (debug) {
                fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
            }
        }
    }

    std::vector<llama_token> embd_inp;
    if ( !params.prompt.empty() || session_tokens.empty() ) {
        // Add a space in front of the first character to match OG llama tokenizer behavior
        params.prompt.insert(0, 1, ' ');

        embd_inp = ::llama_tokenize(ctx, params.prompt, true);
    } else {
        embd_inp = session_tokens;
    }

    // Tokenize negative prompt
    std::vector<llama_token> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;
    if (ctx_guidance) {
        params.cfg_negative_prompt.insert(0, 1, ' ');
        guidance_inp = ::llama_tokenize(ctx_guidance, params.cfg_negative_prompt, true);

        std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, true);
        original_prompt_len = original_inp.size();
        guidance_offset = (int)guidance_inp.size() - original_prompt_len;
    }


    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (debug) {
            if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
                fprintf(stderr, "%s: using full prompt from session file\n", __func__);
            } else if (n_matching_session_tokens >= embd_inp.size()) {
                fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
            } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
                fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
            } else {
                fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
            }
        }
    }
    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() &&
            session_tokens.size() > embd_inp.size()) {
        session_tokens.resize(embd_inp.size() - 1);
    }
    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    }

    if (debug && ctx_guidance) {
            fprintf(stderr, "\n");
            fprintf(stderr, "%s: negative prompt: '%s'\n", __func__, params.cfg_negative_prompt.c_str());
            fprintf(stderr, "%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
            for (int i = 0; i < (int) guidance_inp.size(); i++) {
                fprintf(stderr, "%6d -> '%s'\n", guidance_inp[i], llama_token_to_str(ctx, guidance_inp[i]));
            }
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    grammar_parser::parse_state parsed_grammar;
    llama_grammar *             grammar = NULL;
    if (!params.grammar.empty()) {
        parsed_grammar = grammar_parser::parse(params.grammar.c_str());
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty()) {
            return 1;
        }
        fprintf(stderr, "%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        fprintf(stderr, "\n");

        {
            auto it = params.logit_bias.find(llama_token_eos());
            if (it != params.logit_bias.end() && it->second == -INFINITY) {
                fprintf(stderr,
                    "%s: warning: EOS token is disabled, which will cause most grammars to fail\n", __func__);
            }
        }

        std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
        grammar = llama_grammar_init(
            grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }


    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
    int n_past     = 0;
    int n_remain   = params.n_predict;
    int n_consumed = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;
    std::string res = "";

    // do one empty run to warm up the model
    {
        const std::vector<llama_token> tmp = { llama_token_bos(), };
        llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        llama_reset_timings(ctx);
    }
    
    while (n_remain != 0) {
               // predict
        if (embd.size() > 0) {
            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            auto max_embd_size = n_ctx - 4;
            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int)embd.size() > max_embd_size) {
                auto skipped_tokens = embd.size() - max_embd_size;
                printf("<<input too long: skipped %zu token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                embd.resize(max_embd_size);
            }
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() + std::max<int>(0, guidance_offset) > n_ctx) {
                const int n_left = n_past - params.n_keep;

                // always keep the first token - BOS
                n_past = std::max(1, params.n_keep);
                n_past_guidance = std::max(1, params.n_keep + guidance_offset);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                // stop saving session if we run out of context
                path_session.clear();

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", llama_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
            }


            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

 // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always

            if (ctx_guidance) {
                int input_size = 0;
                llama_token* input_buf = NULL;

                if (n_past_guidance < (int) guidance_inp.size()) {
                    // Guidance context should have the same data with these modifications:
                    //
                    // * Replace the initial prompt
                    // * Shift everything by guidance_offset
                    embd_guidance = guidance_inp;
                    if (embd.begin() + original_prompt_len < embd.end()) {
                        embd_guidance.insert(
                            embd_guidance.end(),
                            embd.begin() + original_prompt_len,
                            embd.end()
                        );
                    }

                    input_buf = embd_guidance.data();
                    input_size = embd_guidance.size();
                    //fprintf(stderr, "\n---------------------\n");
                    //for (int i = 0; i < (int) embd_guidance.size(); i++) {
                        //fprintf(stderr, "%s", llama_token_to_str(ctx, embd_guidance[i]));
                    //}
                    //fprintf(stderr, "\n---------------------\n");
                } else {
                    input_buf = embd.data();
                    input_size = embd.size();
                }

                for (int i = 0; i < input_size; i += params.n_batch) {
                    int n_eval = std::min(input_size - i, params.n_batch);
                    if (llama_eval(ctx_guidance, input_buf + i, n_eval, n_past_guidance, params.n_threads)) {
                        fprintf(stderr, "%s : failed to eval\n", __func__);
                        return 1;
                    }

                    n_past_guidance += n_eval;
                }
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

            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();
        embd_guidance.clear();

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

            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            llama_token id = 0;

            {
                auto logits  = llama_get_logits(ctx);
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
                if (grammar != NULL) {
                    llama_sample_grammar(ctx, &candidates_p, grammar);
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
                if (grammar != NULL) {
                    llama_grammar_accept_token(ctx, grammar, id);
                }
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

     // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(ctx, id);
                }

                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos) {
                        break;
                    }
                }
            }
        }
      
        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
                break;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        if (debug) {
            fprintf(stderr, "\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        }
        llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

end:
#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    if (debug) {
        llama_print_timings(ctx);
        llama_reset_timings(ctx);
    }
    if (grammar != NULL) {
        llama_grammar_free(grammar);
    }

    llama_backend_free();

    strcpy(result, res.c_str()); 
    return 0;
}

void llama_binding_free_model(void *state_ptr) {
    llama_context* ctx = (llama_context*) state_ptr;
    delete &ctx->model;
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

int load_state(void *ctx, char *statefile, char*modes) {
    llama_context* state = (llama_context*) ctx;
const llama_context* constState = static_cast<const llama_context*>(state);
    const size_t state_size = llama_get_state_size(state);
    uint8_t * state_mem = new uint8_t[state_size];

  {
        FILE *fp_read = fopen(statefile, modes);
        if (state_size != llama_get_state_size(constState)) {
            fprintf(stderr, "\n%s : failed to validate state size\n", __func__);
            return 1;
        }

        const size_t ret = fread(state_mem, 1, state_size, fp_read);
        if (ret != state_size) {
            fprintf(stderr, "\n%s : failed to read state\n", __func__);
            return 1;
        }

        llama_set_state_data(state, state_mem);  // could also read directly from memory mapped file
        fclose(fp_read);
    }

    return 0;
}

void save_state(void *ctx, char *dst, char*modes) {
    llama_context* state = (llama_context*) ctx;

    const size_t state_size = llama_get_state_size(state);
    uint8_t * state_mem = new uint8_t[state_size];

    // Save state (rng, logits, embedding and kv_cache) to file
    {
        FILE *fp_write = fopen(dst, modes);
        llama_copy_state_data(state, state_mem); // could also copy directly to memory mapped file
        fwrite(state_mem, 1, state_size, fp_write);
        fclose(fp_write);
    }
}

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, bool memory_f16, int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                             float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap,
                             const char *maingpu,const char *tensorsplit , bool prompt_cache_ro, const char *grammar,
                             float rope_freq_base, float rope_freq_scale, float negative_prompt_scale, const char* negative_prompt
                             ) {
    gpt_params* params = new gpt_params;
    params->seed = seed;
    params->n_threads = threads;
    params->n_predict = tokens;
    params->repeat_last_n = repeat_last_n;
    params->prompt_cache_ro = prompt_cache_ro;
    params->top_k = top_k;
    params->top_p = top_p;
    params->memory_f16 = memory_f16;
    params->temp = temp;
    params->use_mmap = mmap;
    params->use_mlock = mlock;
    params->repeat_penalty = repeat_penalty;
    params->n_batch = n_batch;
    params->n_keep = n_keep;
    params->grammar = std::string(grammar);
    params->rope_freq_base = rope_freq_base;
    params->rope_freq_scale = rope_freq_scale;
    params->cfg_scale = negative_prompt_scale;
    params->cfg_negative_prompt = std::string(negative_prompt);

    if (maingpu[0] != '\0') { 
        params->main_gpu = std::stoi(maingpu);
    }

    if (tensorsplit[0] != '\0') { 
        std::string arg_next = tensorsplit;
            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

            for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
                if (i < split_arg.size()) {
                    params->tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params->tensor_split[i] = 0.0f;
                }
            }  
    }

    params->prompt_cache_all = prompt_cache_all;
    params->path_prompt_cache = session_file;

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

void* load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch, const char *maingpu, const char *tensorsplit, bool numa) {
   return load_binding_model(fname, n_ctx, n_seed, memory_f16, mlock, embeddings, mmap, low_vram, n_gpu_layers, n_batch, maingpu, tensorsplit, numa);
}

/*

Currently we hard patch the following functions to common.cpp and common.h into the llama library due to a bug into the nvcc/gcc compiler. 
It seems that copying by value lead to a misalignment of structure and copy - resulting in a mixed up values that we pass by.

See also: https://github.com/ggerganov/llama.cpp/pull/1902
Keeping them here in sync to generate again patches if needed.

common.h:

struct llama_state {
    llama_context * ctx;
    llama_model * model;
};

common.cpp:
void* load_binding_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch, const char *maingpu, const char *tensorsplit, bool numa) {
    // load the model
    gpt_params lparams;
    llama_model * model;
    llama_state * state;
    state = new llama_state;
    llama_context * ctx;
    lparams.n_ctx      = n_ctx;
    lparams.seed       = n_seed;
    lparams.memory_f16     = memory_f16;
    lparams.embedding  = embeddings;
    lparams.use_mlock  = mlock;
    lparams.n_gpu_layers = n_gpu_layers;
    lparams.use_mmap = mmap;
    lparams.low_vram = low_vram;
    lparams.model = std::string(fname);

    if (maingpu[0] != '\0') { 
        lparams.main_gpu = std::stoi(maingpu);
    }

    if (tensorsplit[0] != '\0') { 
        std::string arg_next = tensorsplit;
            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

            for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
                if (i < split_arg.size()) {
                    lparams.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    lparams.tensor_split[i] = 0.0f;
                }
            }  
    }

    lparams.n_batch      = n_batch;

    llama_backend_init(numa);

    std::tie(model, ctx) = llama_init_from_gpt_params(&lparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return nullptr;
    }
    state->ctx = ctx;
    state->model= model;
    return state;
}

*/