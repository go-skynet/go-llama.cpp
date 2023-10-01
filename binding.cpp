#include "common.h"
#include "llama.h"

#include "binding.h"
#include "grammar-parser.h"
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
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
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;
    gpt_params params = *params_p;

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }
    
    // no need for a rng
    // std::mt19937 rng(params.seed);
  
    int n_past = 0;

    const bool add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM;
    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos);


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
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;
    gpt_params params = *params_p;
 
    for (int i = 0; i < tokenSize; i++) {
        auto token_str = llama_token_to_piece(ctx, tokens[i]);
        std::vector<std::string> my_vector;
        std::string str_token(token_str); // create a new std::string from the char*
        params_p->prompt += str_token;
    }

  return get_embeddings(params_ptr,state_pr,res_embeddings);
}

int eval(void* params_ptr,void* state_pr,char *text) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;

    auto n_past = 0;
    auto last_n_tokens_data = std::vector<llama_token>(params_p->repeat_last_n, 0);

    auto tokens = std::vector<llama_token>(params_p->n_ctx);
    std::string str = std::string(text);
    auto n_prompt_tokens = llama_tokenize(ctx, str.data(), str.length(), tokens.data(), tokens.size(), true);

    if (n_prompt_tokens < 1) {
        fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
        return 1;
    }

    // evaluate prompt
    return llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past, params_p->n_threads);
}

static llama_context ** g_ctx;
static gpt_params               * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;

int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;

    gpt_params params = *params_p;
    g_params = &params;
    const int n_ctx = llama_n_ctx(ctx);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    // no need for a rng
    // std::mt19937 rng(params.seed);

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
            // no need to set the seed here --- we'll always set it later
            // llama_set_rng_seed(ctx, params.seed);
            if (debug) {
                fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
            }
        } else {
            if (debug) {
                fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
            }
        }
    }
    const bool add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM;

    std::vector<llama_token> embd_inp;
    if ( !params.prompt.empty() || session_tokens.empty() ) {
        embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos);
    } else {
        embd_inp = session_tokens;
    }

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(ctx));
    }
    // Tokenize negative prompt
    std::vector<llama_token> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;
    if (ctx_guidance) {
        guidance_inp = ::llama_tokenize(ctx_guidance, params.cfg_negative_prompt, add_bos);
        std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, add_bos);
        original_prompt_len = original_inp.size();
        guidance_offset = (int)guidance_inp.size() - original_prompt_len;
    }


    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size() > 0) {
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
                fprintf(stderr, "%6d -> '%s'\n", guidance_inp[i], llama_token_to_piece(ctx, guidance_inp[i]).c_str());
            }
    }

    struct llama_grammar * grammar = NULL;
    grammar_parser::parse_state parsed_grammar;
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
            auto it = params.logit_bias.find(llama_token_eos(ctx));
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
    std::vector<llama_token> last_tokens(n_ctx);
    std::fill(last_tokens.begin(), last_tokens.end(), 0);

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;

    // the first thing we will do is to output the prompt, so set color accordingly

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;
    const int n_vocab = llama_n_vocab(ctx);
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    std::string res = "";

    {
        const std::vector<llama_token> tmp = { llama_token_bos(ctx), };
        llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        llama_reset_timings(ctx);
    }
    
    // set the seed before actually predicting
    llama_set_rng_seed(ctx, params.seed);

    while (n_remain != 0) {
               // predict
        if (embd.size() > 0) {
            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            auto max_embd_size = n_ctx - 4;
            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int)embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
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

                // insert n_left/2 tokens at the start of embd from last_tokens
                embd.insert(embd.begin(), last_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_tokens.end() - embd.size());

                // stop saving session if we run out of context
                path_session.clear();
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
                        //fprintf(stderr, "%s", llama_token_to_piece(ctx, embd_guidance[i]));
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
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            const llama_token id = llama_sample_token_binding(ctx, ctx_guidance, grammar, params_p, last_tokens, candidates);
            //const llama_token id = llama_sample_token(ctx, ctx_guidance, grammar, params, last_tokens, candidates);

            last_tokens.erase(last_tokens.begin());
            last_tokens.push_back(id);

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;


            // call the token callback, no need to check if one is actually registered, that will
            // be handled on the Go side.
            auto token_str = llama_token_to_piece(ctx, id);
            if (!tokenCallback(state_pr, (char*)token_str.c_str())) {
                break;
            }
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_tokens.erase(last_tokens.begin());
                last_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        for (auto id : embd) {
            const std::string token_str = llama_token_to_piece(ctx, id);
            if (debug) {
              printf("%s", token_str.c_str());
            }

            if (embd.size() > 1) {
                input_tokens.push_back(id);
            } else {
                output_tokens.push_back(id);
                output_ss << token_str;
            }
            res += llama_token_to_piece(ctx, id).c_str();
        }

     // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_tokens) {
                    last_output += llama_token_to_piece(ctx, id);
                }

                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        is_antiprompt = true;
                        break;
                    }
                }
            }
        }

        // found antiprompt
        if (is_antiprompt) {
            break;
        }
      
        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos(ctx)) {
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

// this is a bit of a hack now - ideally this should be in the predict function
// and be transparent to the caller, however this now maps 1:1 (mostly) the upstream implementation
// Note: both model have to be loaded with perplexity "true" to enable all logits
int speculative_sampling(void* params_ptr, void* target_model, void* draft_model, char* result, bool debug) {

    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_binding_state* target_model_state = (llama_binding_state*) target_model;
    llama_binding_state* draft_model_state = (llama_binding_state*) draft_model;

    gpt_params params = *params_p;
    llama_context * ctx_tgt = target_model_state->ctx;
    llama_context * ctx_dft  = draft_model_state->ctx;

    llama_model * model_tgt = target_model_state->model;
    llama_model * model_dft = draft_model_state->model;

    std::string res = "";

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx_tgt, params.prompt, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }
  
    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    llama_eval(ctx_tgt,  inp.data(), int(inp.size() - 1), 0, params.n_threads);
    llama_eval(ctx_tgt, &inp.back(),      1, inp.size() - 1, params.n_threads);
    llama_eval(ctx_dft,  inp.data(),     int(inp.size()), 0, params.n_threads);

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    const int n_ctx   = llama_n_ctx(ctx_tgt);
    const int n_vocab = llama_n_vocab(ctx_tgt);
    //GGML_ASSERT(n_vocab == llama_n_vocab(ctx_dft));

    // how many tokens to draft each time
    const int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    std::vector<llama_token> drafted;

    std::vector<llama_token> last_tokens(n_ctx);
    std::fill(last_tokens.begin(), last_tokens.end(), 0);

    for (auto & id : inp) {
        last_tokens.erase(last_tokens.begin());
        last_tokens.push_back(id);
    }

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    // used to determine end of generation
    bool has_eos = false;

    // grammar stuff
    struct llama_grammar * grammar_dft = NULL;
    struct llama_grammar * grammar_tgt = NULL;

    grammar_parser::parse_state parsed_grammar;

    // if requested - load the grammar, error checking is omitted for brevity
    if (!params.grammar.empty()) {
        parsed_grammar = grammar_parser::parse(params.grammar.c_str());
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty()) {
            return 1;
        }

        std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
        grammar_tgt = llama_grammar_init(grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }

    const auto t_dec_start = ggml_time_us();

    while (true) {
        int i_dft = 0;
        while (true) {
            // sample from the target model

            // const llama_token id = llama_sample_token(ctx_tgt, NULL, grammar_tgt, params, last_tokens, candidates, i_dft);
            const llama_token id = llama_sample_token_binding(ctx_tgt, NULL, grammar_tgt, params_p, last_tokens, candidates, i_dft);
            // remember which tokens were sampled - used for repetition penalties during sampling
            last_tokens.erase(last_tokens.begin());
            last_tokens.push_back(id);

            //LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, last_tokens));

            const std::string token_str = llama_token_to_piece(ctx_tgt, id);
            if (!tokenCallback(draft_model, (char*)token_str.c_str())) {
                break;
            }       
            res += token_str.c_str();
        
            if (id == llama_token_eos(ctx_tgt)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the draft matches the target
            if (i_dft < (int) drafted.size() && id == drafted[i_dft]) {
                LOG("drafted token %d accepted\n", id);
                ++n_accept;
                ++n_past_tgt;
                ++n_past_dft;
                ++i_dft;

                continue;
            }

            if (i_dft < (int) drafted.size()) {
                LOG("the %dth drafted token (%d, '%s') does not match the sampled target token (%d, '%s') - rejected\n",
                        i_dft, drafted[i_dft], llama_token_to_piece(ctx_dft, drafted[i_dft]).c_str(), id, token_str.c_str());
            } else {
                LOG("out of drafted tokens\n");
            }

            // the drafted token was rejected or we are out of drafted tokens
            llama_eval(ctx_dft, &id, 1, n_past_dft, params.n_threads);
            ++n_past_dft;

            drafted.clear();
            drafted.push_back(id);

            break;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        if (grammar_tgt) {
            if (grammar_dft) {
                llama_grammar_free(grammar_dft);
            }
            grammar_dft = llama_grammar_copy(grammar_tgt);

            LOG("copied target grammar to draft grammar\n");
        }

        // sample n_draft tokens from the draft model using greedy decoding
        int n_past_cur = n_past_dft;
        for (int i = 0; i < n_draft; ++i) {
            float * logits = llama_get_logits(ctx_dft);

            candidates.clear();
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array cur_p = { candidates.data(), candidates.size(), false };

            if (grammar_dft != NULL) {
                llama_sample_grammar(ctx_dft, &cur_p, grammar_dft);
            }

            // computes softmax and sorts the candidates
            llama_sample_softmax(ctx_dft, &cur_p);

            for (int i = 0; i < 3; ++i) {
                LOG(" - draft candidate %d: %d (%.3f)\n", i, cur_p.data[i].id, cur_p.data[i].p);
            }

            // TODO: better logic?
            if (cur_p.data[0].p < 2*cur_p.data[1].p) {
                LOG("stopping drafting, probability too low: %.3f < 2*%.3f\n", cur_p.data[0].p, cur_p.data[1].p);
                break;
            }

            // drafted token
            const llama_token id = cur_p.data[0].id;

            drafted.push_back(id);
            ++n_drafted;

            // no need to evaluate the last drafted token, since we won't use the result
            if (i == n_draft - 1) {
                break;
            }

            // evaluate the drafted token on the draft model
            llama_eval(ctx_dft, &drafted.back(), 1, n_past_cur, params.n_threads);
            ++n_past_cur;

            if (grammar_dft != NULL) {
                llama_grammar_accept_token(ctx_dft, grammar_dft, id);
            }
        }

        // evaluate the target model on the drafted tokens
        llama_eval(ctx_tgt, drafted.data(), drafted.size(), n_past_tgt, params.n_threads);
        ++n_past_tgt;
        
        // the first token is always proposed by the traget model before the speculation loop
        drafted.erase(drafted.begin());
    }
    if (debug) {
        auto t_dec_end = ggml_time_us();

        LOG_TEE("\n\n");

        LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
        LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

        // TODO: make sure these numbers are computed correctly
        LOG_TEE("\n");
        LOG_TEE("n_draft   = %d\n", n_draft);
        LOG_TEE("n_predict = %d\n", n_predict);
        LOG_TEE("n_drafted = %d\n", n_drafted);
        LOG_TEE("n_accept  = %d\n", n_accept);
        LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

        LOG_TEE("\ndraft:\n");
        llama_print_timings(ctx_dft);

        LOG_TEE("\ntarget:\n");
        llama_print_timings(ctx_tgt);

        fprintf(stderr, "\n\n");
    }
    if (grammar_dft != NULL) {
        llama_grammar_free(grammar_dft);
        llama_grammar_free(grammar_tgt);
    }
    strcpy(result, res.c_str()); 
    return 0;
}

void llama_binding_free_model(void *state_ptr) {
    llama_binding_state* ctx = (llama_binding_state*) state_ptr;
    llama_free(ctx->ctx);
    delete ctx->model;
}

void llama_free_params(void* params_ptr) {
    gpt_params* params = (gpt_params*) params_ptr;
    delete params;
}

int llama_tokenize_string(void* params_ptr, void* state_pr, int* result) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;

    const bool add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM;

    return llama_tokenize(ctx, params_p->prompt.data(), params_p->prompt.length(), result, params_p->n_ctx, add_bos);
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
                             float rope_freq_base, float rope_freq_scale, float negative_prompt_scale, const char* negative_prompt, int n_draft) {
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
    params->n_draft = n_draft;
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
        params->ignore_eos = true;
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

void* load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch, const char *maingpu, const char *tensorsplit, bool numa, float rope_freq_base, float rope_freq_scale, bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity) {
   return load_binding_model(fname, n_ctx, n_seed, memory_f16, mlock, embeddings, mmap, low_vram, n_gpu_layers, n_batch, maingpu, tensorsplit, numa, rope_freq_base, rope_freq_scale, mul_mat_q, lora, lora_base, perplexity);
}

/*

Currently we hard patch the following functions to common.cpp and common.h into the llama library due to a bug into the nvcc/gcc compiler. 
It seems that copying by value lead to a misalignment of structure and copy - resulting in a mixed up values that we pass by.

See also: https://github.com/ggerganov/llama.cpp/pull/1902
Keeping them here in sync to generate again patches if needed.

common.h:

struct llama_binding_state {
    llama_context * ctx;
    llama_model * model;
};

void* load_binding_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch, const char *maingpu, const char *tensorsplit, bool numa,  float rope_freq_base, float rope_freq_scale, bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity);

llama_token llama_sample_token_binding(
                  struct llama_context * ctx,
                  struct llama_context * ctx_guidance,
                  struct llama_grammar * grammar,
               const struct gpt_params * g_params,
        const std::vector<llama_token> & last_tokens,
         std::vector<llama_token_data> & candidates,
                                   int   idx = 0);

common.cpp:

gpt_params* create_gpt_params(const std::string& fname,const std::string& lora,const std::string& lora_base) {
   gpt_params* lparams = new gpt_params;
    fprintf(stderr, "%s: loading model %s\n", __func__, fname.c_str());

    // Initialize the 'model' member with the 'fname' parameter
    lparams->model = fname;
    lparams->lora_base = lora_base;
    lparams->lora_adapter = lora;
    if (lparams->lora_adapter.empty()) {
        lparams->use_mmap = false;
    }
    return lparams;
}

gpt_params* create_gpt_params_cuda(const std::string& fname) {
   gpt_params* lparams = new gpt_params;
    fprintf(stderr, "%s: loading model %s\n", __func__, fname.c_str());

    // Initialize the 'model' member with the 'fname' parameter
    lparams->model = fname;
    return lparams;
}

void* load_binding_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch, const char *maingpu, const char *tensorsplit, bool numa,  float rope_freq_base, float rope_freq_scale, bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity) {
    // load the model
    gpt_params * lparams;
// Temporary workaround for https://github.com/go-skynet/go-llama.cpp/issues/218
#ifdef GGML_USE_CUBLAS
    lparams = create_gpt_params_cuda(fname);
#else
    lparams = create_gpt_params(fname, lora, lora_base);
#endif
    llama_model * model;
    llama_binding_state * state;
    state = new llama_binding_state;
    llama_context * ctx;
    lparams->n_ctx      = n_ctx;
    lparams->seed       = n_seed;
    lparams->memory_f16     = memory_f16;
    lparams->embedding  = embeddings;
    lparams->use_mlock  = mlock;
    lparams->n_gpu_layers = n_gpu_layers;
    lparams->perplexity = perplexity;
    lparams->use_mmap = mmap;

    lparams->low_vram = low_vram;
    if (rope_freq_base != 0.0f) {
        lparams->rope_freq_base = rope_freq_base;
    } else {
        lparams->rope_freq_base = 10000.0f;
    }

    if (rope_freq_scale != 0.0f) {
        lparams->rope_freq_scale = rope_freq_scale;
    } else {
        lparams->rope_freq_scale =  1.0f;
    }

    lparams->model = fname;
    if (maingpu[0] != '\0') { 
        lparams->main_gpu = std::stoi(maingpu);
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
                    lparams->tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    lparams->tensor_split[i] = 0.0f;
                }
            }  
    }

    lparams->n_batch      = n_batch;

    llama_backend_init(numa);

    std::tie(model, ctx) = llama_init_from_gpt_params(*lparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return nullptr;
    }
    state->ctx = ctx;
    state->model= model;
    return state;
}

// Note: the only difference here is passing params as a pointer and avoid copy-by-value
// We stick to another function to avoid patching all the llama.cpp code
// We need the function to be in the common.o object, as using it in the binding does not make effect.
llama_token llama_sample_token_binding(
                  struct llama_context * ctx,
                  struct llama_context * ctx_guidance,
                  struct llama_grammar * grammar,
               const struct gpt_params * g_params,  // NOTE: this is our patch
        const std::vector<llama_token> & last_tokens,
         std::vector<llama_token_data> & candidates,
                                   int   idx) {

   
    struct gpt_params params = *g_params;  // NOTE: this is our patch
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(ctx);

    const float   temp            = params.temp;
    const int32_t top_k           = params.top_k <= 0 ? n_vocab : params.top_k;
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

    float * logits = llama_get_logits(ctx) + idx * n_vocab;

    // Apply params.logit_bias map
    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
        logits[it->first] += it->second;
    }

    candidates.clear();
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array cur_p = { candidates.data(), candidates.size(), false };

    if (ctx_guidance) {
        llama_sample_classifier_free_guidance(ctx, &cur_p, ctx_guidance, params.cfg_scale);
    }

    // apply penalties
    if (!last_tokens.empty()) {
        const float nl_logit = logits[llama_token_nl(ctx)];
        const int last_n_repeat = std::min(std::min((int)last_tokens.size(), repeat_last_n), n_ctx);

        llama_sample_repetition_penalty(ctx, &cur_p,
                last_tokens.data() + last_tokens.size() - last_n_repeat,
                last_n_repeat, repeat_penalty);
        llama_sample_frequency_and_presence_penalties(ctx, &cur_p,
                last_tokens.data() + last_tokens.size() - last_n_repeat,
                last_n_repeat, alpha_frequency, alpha_presence);

        if (!penalize_nl) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == llama_token_nl(ctx)) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    if (grammar != NULL) {
        llama_sample_grammar(ctx, &cur_p, grammar);
    }

    if (temp <= 0) {
        // Greedy sampling
        id = llama_sample_token_greedy(ctx, &cur_p);
    } else {
        if (mirostat == 1) {
            static float mirostat_mu = 2.0f * mirostat_tau;
            const int mirostat_m = 100;
            llama_sample_temperature(ctx, &cur_p, temp);
            id = llama_sample_token_mirostat(ctx, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
        } else if (mirostat == 2) {
            static float mirostat_mu = 2.0f * mirostat_tau;
            llama_sample_temperature(ctx, &cur_p, temp);
            id = llama_sample_token_mirostat_v2(ctx, &cur_p, mirostat_tau, mirostat_eta, &mirostat_mu);
        } else {
            // Temperature sampling
            llama_sample_top_k      (ctx, &cur_p, top_k, 1);
            llama_sample_tail_free  (ctx, &cur_p, tfs_z, 1);
            llama_sample_typical    (ctx, &cur_p, typical_p, 1);
            llama_sample_top_p      (ctx, &cur_p, top_p, 1);
            llama_sample_temperature(ctx, &cur_p, temp);

            {
                const int n_top = 10;
                LOG("top %d candidates:\n", n_top);

                for (int i = 0; i < n_top; i++) {
                    const llama_token id = cur_p.data[i].id;
                    LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(ctx, id).c_str(), cur_p.data[i].p);
                }
            }

            id = llama_sample_token(ctx, &cur_p);

            LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx, id).c_str());
        }
    }
    // printf("`%d`", candidates_p.size);

    if (grammar != NULL) {
        llama_grammar_accept_token(ctx, grammar, id);
    }

    return id;
}

*/
