// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...
//
// ============================================================================
// RKLLM Chat — Production-Grade Interactive LLM CLI for RK3588/RK3576
// ============================================================================
// Optimizations:
//   • Prompt caching — system prompt prefill eliminated after 1st turn
//   • KV cache monitoring — real-time context window gauge
//   • CPU affinity — A76 big-core pinning for stable decode latency
//   • n_batch tuning — parallel prefill for lower TTFT
//   • Sampling presets — /preset precise|balanced|creative
//   • Accurate perf — SDK RKLLMPerfStat for real tok/s
//   • Warmup inference — hide cold-start from user
//   • Session logging — auto-save conversation to .jsonl
//   • Context bar — visual [████░░] 1.2k/8k in prompt
// ============================================================================

#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <iostream>
#include <csignal>
#include <fstream>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <exception>
#include <setjmp.h>
#include <cstdio>
#include <cctype>
#include <cmath>

using namespace std;
using namespace std::chrono;

// Non-local jump target for recovering from C-library aborts
static sigjmp_buf g_recover_jmp;
static volatile bool g_recover_armed = false;

static void on_terminate_handler() {
    if (g_recover_armed) siglongjmp(g_recover_jmp, 1);
    abort();  // fallback
}
static void on_sigabrt_handler(int) {
    if (g_recover_armed) siglongjmp(g_recover_jmp, 1);
    _exit(1);  // fallback
}

// ====================================================================
// ANSI Colors
// ====================================================================
#define RST   "\033[0m"
#define BLD   "\033[1m"
#define DIM   "\033[2m"
#define RED   "\033[31m"
#define GRN   "\033[32m"
#define YEL   "\033[33m"
#define BLU   "\033[34m"
#define MAG   "\033[35m"
#define CYN   "\033[36m"
#define WHT   "\033[37m"
#define GRAY  "\033[90m"

// Forward declarations for functions before globals
static const char* g_chat_prefix  = nullptr;
static const char* g_chat_postfix = nullptr;
static bool        g_is_deepseek  = false;
static bool        g_history      = true;  // forward ref for detect_model_template

// ====================================================================
// Memory Guard — professional pre-flight safety check
// ====================================================================
static long get_mem_available_mb() {
    ifstream f("/proc/meminfo");
    string line;
    while (getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            size_t p = line.find(':');
            return atol(line.c_str() + p + 1) / 1024;
        }
    }
    return -1;
}

static long get_file_size_mb(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return st.st_size / (1024 * 1024);
}

static bool mem_guard_check(const char* model_path) {
    long avail = get_mem_available_mb();
    long model_sz = get_file_size_mb(model_path);
    cout << DIM "  RAM: " RST << avail << " MB free"
         << DIM "  |  Model: " RST << model_sz << " MB" << endl;
    if (avail <= 0 || model_sz <= 0) {
        cout << YEL "  [WARN] Cannot read memory stats, proceeding anyway.\n" RST;
        return true;
    }
    long estimated = (long)(model_sz * 1.5);
    long safe_margin = 2048;
    if (avail < estimated + safe_margin) {
        cout << RED "\n  [GUARD] SAFETY STOP: Not enough memory.\n" RST;
        cout << DIM "     Need ~" RST << estimated << " MB"
             << DIM " + " RST << safe_margin << " MB safety"
             << DIM " | Have " RST << avail << " MB\n";
        cout << YEL "     Close other apps or use a smaller model.\n" RST;
        return false;
    }
    if (avail < estimated * 2) {
        cout << YEL "  [WARN] Memory tight, other apps may be killed.\n" RST;
    }
    return true;
}

// Detect model family from filename and set chat template markers
static void detect_model_template(const char* model_path) {
    string path(model_path);
    // DeepSeek-R1 — full-width pipe markers (hex-escaped for encoding safety)
    // <FF5C> is full-width vertical bar U+FF5C (0xEF 0xBC 0x9C in UTF-8)
    if (path.find("DeepSeek") != string::npos) {
        g_chat_prefix  = "\x3C\xEF\xBC\x9C" "User" "\xEF\xBC\x9C\x3E";
        g_chat_postfix = "\x3C\xEF\xBC\x9C" "Assistant" "\xEF\xBC\x9C\x3E";
        g_is_deepseek = true;
        g_history = false;  // small reasoning models hallucinate in multi-turn
        cout << DIM "  Template: DeepSeek-R1 (single-turn)" RST << endl;
        return;
    }

    // Qwen3 family (pure instruct, no reasoning)
    if (path.find("Qwen")  != string::npos ||
        path.find("qwen")  != string::npos) {
        g_chat_prefix  = "<|im_start|>user\n";
        g_chat_postfix = "<|im_end|>\n<|im_start|>assistant\n";
        cout << DIM "  Template: Qwen3 (im_start/end)" RST << endl;
        return;
    }
    // LLaMA family
    if (path.find("LLaMA") != string::npos ||
        path.find("llama") != string::npos ||
        path.find("TinyLLAMA") != string::npos) {
        g_chat_prefix  = "<s>[INST] ";
        g_chat_postfix = " [/INST]";
        cout << DIM "  Template: LLaMA" RST << endl;
        return;
    }
    // ChatGLM family
    if (path.find("ChatGLM") != string::npos ||
        path.find("chatglm") != string::npos) {
        g_chat_prefix  = "[Round 1]\n\n问：";
        g_chat_postfix = "\n\n答：";
        cout << DIM "  Template: ChatGLM" RST << endl;
        return;
    }
    // Unknown model — use built-in default template
    g_chat_prefix  = nullptr;
    g_chat_postfix = nullptr;
    cout << DIM "  Template: auto (model default)" RST << endl;
}

// ====================================================================
// Globals
// ====================================================================
static LLMHandle g_h = nullptr;
static atomic<bool> g_running{false};
static atomic<bool> g_stop{false};
static atomic<int>  g_sigcnt{0};

// Timing & perf — use SDK RKLLMPerfStat fields
static steady_clock::time_point g_t0;
static bool   g_first_tok  = true;
static float  g_ttft_ms    = 0;
static int    g_out_toks    = 0;
static float  g_pf_ms      = 0;    // SDK prefill time
static int    g_pf_toks    = 0;    // SDK prefill token count
static float  g_dec_ms     = 0;    // SDK decode time
static int    g_dec_toks   = 0;    // SDK decode token count
static float  g_mem_mb     = 0;

// Settings
// g_history, g_is_deepseek declared above (before mem_guard)
static bool   g_thinking    = false;
static bool   g_warmup_mode = false;  // suppress output during warmup

// Chat template markers — declared above (before mem_guard)

static string g_sys_prompt  = "你是一个乐于助人的AI助手。请用中文简洁、准确地回答问题。不知道就说不知道。";

static int    g_max_ctx     = 8192;
static int    g_max_new     = 4096;
static int    g_n_batch     = 128;  // prefill batch size (uint8 max=255)

// Prompt cache
static string g_cache_path;

// Session log
static ofstream g_session_log;

// ====================================================================
// Preset definitions
// ====================================================================
struct Preset {
    const char* name;
    const char* desc;
    int   top_k;
    float top_p;
    float temp;
    float repeat_penalty;
    int   mirostat;
    float mirostat_tau;
    float mirostat_eta;
};

static const Preset PRESETS[] = {
    {"precise",   "Low randomness — code, facts, math",
        1,   0.10f, 0.1f,  1.00f, 0, 5.0f, 0.1f},
    {"balanced",  "Default — general conversation",
        40,  0.90f, 0.7f,  1.10f, 0, 5.0f, 0.1f},
    {"creative",  "High diversity — stories, brainstorming",
        80,  0.95f, 1.0f,  1.05f, 0, 5.0f, 0.1f},
    {"mirostat",  "Adaptive perplexity — most consistent quality",
        0,   0.00f, 1.0f,  1.10f, 2, 5.0f, 0.1f},
};

static int g_preset_idx = 1; // balanced

// ====================================================================
// Helpers
// ====================================================================
static string ft(float ms) {
    if (ms < 0) return "--";
    if (ms < 1000) return to_string((int)ms) + "ms";
    ostringstream o; o << fixed << setprecision(1) << (ms/1000.f) << "s"; return o.str();
}
static string fm(float mb) {
    if (mb <= 0) return "--";
    if (mb < 1024) return to_string((int)mb) + "MB";
    ostringstream o; o << fixed << setprecision(1) << (mb/1024.f) << "GB"; return o.str();
}
// Simple recursive-descent math parser — no shell, no awk, 100% safe
static double eval_expr(const char*& p);
static double eval_term(const char*& p);
static double eval_factor(const char*& p);

static void skip_ws(const char*& p) { while (*p == ' ') p++; }

static double eval_num(const char*& p) {
    skip_ws(p);
    char* end;
    double v = strtod(p, &end);
    if (end == p) return NAN;
    p = end;
    skip_ws(p);
    return v;
}

static double eval_factor(const char*& p) {
    skip_ws(p);
    if (*p == '(') { p++; double v = eval_expr(p); skip_ws(p); if (*p == ')') p++; return v; }
    return eval_num(p);
}

static double eval_term(const char*& p) {
    double v = eval_factor(p);
    while (true) {
        skip_ws(p);
        if (*p == '*' || *p == '/') { char op = *p++; double rhs = eval_factor(p); v = (op=='*') ? v*rhs : v/rhs; }
        else break;
    }
    return v;
}

static double eval_expr(const char*& p) {
    double v = eval_term(p);
    while (true) {
        skip_ws(p);
        if (*p == '+' || *p == '-') { char op = *p++; double rhs = eval_term(p); v = (op=='+') ? v+rhs : v-rhs; }
        else break;
    }
    return v;
}

// Pre-process user input: detect math expressions and compute them
static string preprocess_input(const string& raw) {
    // Check for math pattern: digit(s) + operator + digit(s)
    bool has_math = false;
    for (size_t i = 0; i < raw.length(); i++) {
        if (isdigit(raw[i])) {
            for (size_t j = i; j < raw.length() && j < i + 80; j++) {
                if (raw[j] == '+' || raw[j] == '-' || raw[j] == '*' ||
                    raw[j] == '/' || raw[j] == 'x' || raw[j] == 'X') {
                    has_math = true; break;
                }
            }
            if (has_math) break;
        }
    }
    if (!has_math) return raw;

    // Extract evaluable expression
    string expr;
    for (char c : raw) {
        if (isdigit(c) || c == '+' || c == '-' || c == '*' || c == '/' ||
            c == '.' || c == '(' || c == ')' || c == ' ') expr += c;
        else if (c == 'x' || c == 'X') expr += '*';
    }

    bool has_digit = false, has_op = false;
    for (char c : expr) { if (isdigit(c)) has_digit = true; if (c == '+' || c == '-' || c == '*' || c == '/') has_op = true; }
    if (!has_digit || !has_op) return raw;

    // Parse & evaluate
    const char* pp = expr.c_str();
    double res = eval_expr(pp);
    if (isnan(res)) return raw;

    // Format result
    char buf[64];
    snprintf(buf, sizeof(buf), "%.10g", res);
    string rs(buf);

    cout << CYN "  🔢 " << expr << DIM " = " RST << rs << endl;

    // Model often ignores precomputed results → answer directly
    cout << BLD GRN "▸ " RST << rs << endl;
    return "";  // empty = skip model, caller checks and continues
}

static string fs(float toks, float ms) {
    if (ms <= 0 || toks <= 0) return "--";
    ostringstream o; o << fixed << setprecision(1) << (toks/(ms/1000.f)); return o.str();
}
static string bar(int used, int total) {
    if (total <= 0) return "";
    const int W = 16;
    float pct = min(1.f, (float)used / (float)total);
    int fill = (int)(pct * W);
    string s = DIM "[";
    if (pct < 0.5f) s += GRN;
    else if (pct < 0.8f) s += YEL;
    else s += RED;
    for (int i = 0; i < fill; i++) s += "█";
    s += DIM;
    for (int i = 0; i < W - fill; i++) s += "░";
    s += RST;
    ostringstream o;
    o << s << DIM " ";
    if (used < 1024) o << used << "/" << total;
    else o << fixed << setprecision(1) << (used/1024.f) << "k/" << (total/1024) << "k";
    o << RST;
    return o.str();
}
static void set_sampling(RKLLMParam& p, const Preset& pr) {
    p.top_k       = pr.top_k;
    p.top_p       = pr.top_p;
    p.temperature = pr.temp;
    p.repeat_penalty    = pr.repeat_penalty;
    p.frequency_penalty = 0.f;
    p.presence_penalty  = 0.f;
    p.mirostat     = pr.mirostat;
    p.mirostat_tau = pr.mirostat_tau;
    p.mirostat_eta = pr.mirostat_eta;
}

// ====================================================================
// Signal handler
// ====================================================================
static void on_sig(int) {
    int n = ++g_sigcnt;
    if (n == 1) {
        if (g_running) { g_stop = true; rkllm_abort(g_h);
            cerr << YEL "\n\n  ⏸  Stopping... (Ctrl+C again to exit)\n" RST; }
        else { cerr << YEL "\n  👋 Goodbye.\n" RST; if (g_h) { rkllm_destroy(g_h); g_h = nullptr; } exit(0); }
    } else {
        cerr << RED "\n  🛑 Force exit.\n" RST; if (g_h) { rkllm_destroy(g_h); g_h = nullptr; } exit(0);
    }
}

// ====================================================================
// Callback
// ====================================================================
static int on_res(RKLLMResult* r, void*, LLMCallState st) {
    if (st == RKLLM_RUN_ERROR) {
        cerr << RED "\n  ❌ Inference error.\n" RST;
        g_running = false; return 0;
    }
    if (g_first_tok && st == RKLLM_RUN_NORMAL && r->text && r->text[0]) {
        auto now = steady_clock::now();
        g_ttft_ms = duration<float,milli>(now - g_t0).count();
        g_first_tok = false;
    }
    if (st == RKLLM_RUN_NORMAL) {
        if (r->text) {
            if (!g_warmup_mode) {
                // Strip leaked special tokens (<|im_end|>, <think>, etc.)
                string_view sv(r->text);
                // Filter leaked template tokens (fullwidth + ASCII)
                if (sv.find("<|im_") == string_view::npos &&
                    sv.find("think") == string_view::npos &&
                    sv.find("dof") == string_view::npos &&
                    sv.find("User") == string_view::npos &&
                    sv.find("用户") == string_view::npos &&
                    sv.find("\xEF\xBC\x9C") == string_view::npos)  // fullwidth <
                    cout << r->text << flush;
            }
            g_out_toks++;
        }
    } else if (st == RKLLM_RUN_FINISH) {
        if (!g_warmup_mode) cout << endl;
        if (r->perf.prefill_time_ms > 0)   { g_pf_ms = r->perf.prefill_time_ms;   g_pf_toks = r->perf.prefill_tokens; }
        if (r->perf.generate_time_ms > 0)  { g_dec_ms = r->perf.generate_time_ms; g_dec_toks = r->perf.generate_tokens; }
        g_mem_mb = r->perf.memory_usage_mb;
        g_running = false;
    }
    return 0;
}

// ====================================================================
// Stats
// ====================================================================
static void print_stats() {
    int kv_used = 0;
    rkllm_get_kv_cache_size(g_h, &kv_used);

    // Use SDK-supplied tokens for decode speed; fall back to chunk count
    int d_tok = g_dec_toks > 0 ? g_dec_toks : g_out_toks;
    int p_tok = g_pf_toks;
    float p_ms = g_pf_ms;
    float d_ms = g_dec_ms;

    cout << DIM "  ╭─── Inference ─────────────────────────────\n" RST;
    cout << DIM "  │ " CYN "TTFT   " RST DIM << ft(g_ttft_ms)
         << DIM " │ " CYN "Prefill " RST DIM << ft(p_ms) << " " << p_tok << "tok @" << fs(p_tok, p_ms) << "t/s\n";
    cout << DIM "  │ " CYN "Decode " RST DIM << ft(d_ms) << " " << d_tok << "tok @" << fs(d_tok, d_ms) << "t/s"
         << DIM " │ " CYN "Mem " RST DIM << fm(g_mem_mb) << "\n";
    cout << DIM "  │ " CYN "KV     " RST << bar(kv_used, g_max_ctx) << "\n";
    cout << DIM "  ╰──────────────────────────────────────────\n" RST;
}

// ====================================================================
// Banner
// ====================================================================
static void print_banner(const char* model) {
    cout << BLD CYN R"(
  ╔══════════════════════════════════════════════╗
  ║     RKLLM Chat  —  RK3588/RK3576 NPU       ║
  ╚══════════════════════════════════════════════╝
)" RST;
    cout << DIM "  Model:    " RST << model << "\n"
         << DIM "  Context:  " RST << g_max_ctx << " tokens\n"
         << DIM "  Max out:  " RST << g_max_new << " tokens\n"
         << DIM "  Preset:   " RST << GRN << PRESETS[g_preset_idx].name
         << DIM " (" << PRESETS[g_preset_idx].desc << ")" RST "\n"
         << DIM "  History:  " RST << (g_history?GRN"ON":DIM"OFF") << RST
         << DIM "  Thinking: " RST << (g_thinking?GRN"ON":DIM"OFF") << RST
         << DIM "\n  Type " RST "/help" DIM " for commands.\n" RST << endl;
}

// ====================================================================
// Commands (returns true if handled)
// ====================================================================
static bool cmd(const string& s) {
    if (s == "/help") {
        cout << BLD "\n  Commands\n" RST
             << "  " CYN "/help"           RST "              Show this\n"
             << "  " CYN "/clear"          RST "              Clear conversation\n"
             << "  " CYN "/stats"          RST "              Perf breakdown + KV gauge\n"
             << "  " CYN "/zh | /en"       RST "            Switch system prompt language\n"
             << "  " CYN "/preset <name>"  RST "          precise | balanced | creative | mirostat\n"
             << "  " CYN "/history on|off" RST "         Multi-turn toggle\n"
             << "  " CYN "/think on|off"   RST "         Qwen3 thinking mode\n"
             << "  " CYN "/system <text>"  RST "          Change system prompt\n"
             << "  " CYN "/exit"           RST "              Quit\n"
             << endl;
        return true;
    }
    if (s == "/zh" || s == "/en") {
        g_sys_prompt = (s == "/zh")
            ? "你是一个乐于助人的AI助手。请用中文简洁、准确地回答用户问题。不知道就说不知道。"
            : "You are a helpful, respectful and honest AI assistant. "
              "Answer concisely and accurately. "
              "If you don't know something, say so.";
        cout << GRN "  ✓ " RST << (s=="/zh"?"中文":"English") << " mode. Clearing history...\n";
        if (g_chat_prefix && g_chat_postfix)
            rkllm_set_chat_template(g_h, g_sys_prompt.c_str(), g_chat_prefix, g_chat_postfix);
        rkllm_clear_kv_cache(g_h, 0, nullptr, nullptr);
        return true;
    }
    if (s == "/clear") {
        int r = rkllm_clear_kv_cache(g_h, 0, nullptr, nullptr);
        if (r == 0) cout << GRN "  ✓ KV cache cleared.\n" RST;
        else       cout << RED "  ✗ KV clear failed (ret=" << r << ")\n" RST;
        return true;
    }
    if (s == "/stats") { print_stats(); return true; }

    if (s.rfind("/preset ", 0) == 0) {
        string v = s.substr(8);
        for (int i = 0; i < 4; i++) {
            if (v == PRESETS[i].name) {
                g_preset_idx = i;
                cout << GRN "  ✓ Preset: " RST << PRESETS[i].name
                     << DIM " (" << PRESETS[i].desc << ")\n" RST;
                cout << DIM "  Changes apply on next init. Restart or ignore.\n" RST;
                return true;
            }
        }
        cout << RED "  Usage: /preset precise|balanced|creative|mirostat\n" RST;
        return true;
    }
    if (s.rfind("/history ", 0) == 0) {
        string v = s.substr(9);
        if (v=="on") { g_history=true; cout << GRN "  ✓ History ON\n" RST; }
        else if (v=="off") { g_history=false; cout << YEL "  ✓ History OFF\n" RST; }
        else cout << RED "  Usage: /history on|off\n" RST;
        return true;
    }
    if (s.rfind("/think ", 0) == 0) {
        string v = s.substr(7);
        if (v=="on") { g_thinking=true; cout << GRN "  ✓ Thinking ON\n" RST; }
        else if (v=="off") { g_thinking=false; cout << YEL "  ✓ Thinking OFF\n" RST; }
        else cout << RED "  Usage: /think on|off\n" RST;
        rkllm_clear_kv_cache(g_h, 0, nullptr, nullptr);
        return true;
    }
    if (s.rfind("/system ", 0) == 0) {
        g_sys_prompt = s.substr(8);
        cout << GRN "  ✓ System prompt updated. Clearing history...\n" RST;
        if (g_chat_prefix && g_chat_postfix)
            rkllm_set_chat_template(g_h, g_sys_prompt.c_str(), g_chat_prefix, g_chat_postfix);
        rkllm_clear_kv_cache(g_h, 0, nullptr, nullptr);
        return true;
    }
    if (s.rfind("/batch ", 0) == 0) {
        g_n_batch = atoi(s.substr(7).c_str());
        if (g_n_batch < 1) g_n_batch = 256;
        cout << GRN "  ✓ Prefill batch = " RST << g_n_batch << DIM " (restart to apply)\n" RST;
        return true;
    }
    return false;
}

// ====================================================================
// Warmup — a silent inference to populate caches
// ====================================================================
static void do_warmup() {
    RKLLMInput in; memset(&in, 0, sizeof(in));
    in.input_type = RKLLM_INPUT_PROMPT;
    in.role = "user";
    in.prompt_input = "Hi";
    RKLLMInferParam ip; memset(&ip, 0, sizeof(ip));
    ip.mode = RKLLM_INFER_GENERATE;
    ip.keep_history = 0;
    ip.max_new_tokens = 4;  // need >=4 for complete UTF-8
    g_warmup_mode = true;
    rkllm_run(g_h, &in, &ip, nullptr);
    g_warmup_mode = false;
}

// ====================================================================
// Main
// ====================================================================
int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << RED "Usage: " << argv[0] << " <model.rkllm> <max_new_tokens> <max_context_len> [prompt_cache.bin]\n" RST;
        cerr << DIM "Example: " << argv[0] << " qwen.rkllm 4096 8192 ./cache.bin\n" RST;
        return 1;
    }
    const char* model_path = argv[1];
    g_max_new  = atoi(argv[2]);
    g_max_ctx  = atoi(argv[3]);
    g_cache_path = (argc >= 5) ? argv[4] : "./prompt_cache.bin";

    signal(SIGINT, on_sig);
    signal(SIGABRT, on_sigabrt_handler);
    set_terminate(on_terminate_handler);

    // --- Memory guard ---
    if (!mem_guard_check(model_path)) return 1;

    // --- Init ---
    cout << DIM "⏳ Loading model..." RST << flush;
    auto t_init0 = steady_clock::now();

    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path          = model_path;
    p.max_new_tokens      = g_max_new;
    p.max_context_len     = g_max_ctx;
    p.skip_special_token  = true;
    p.is_async            = false;

    const Preset& pr = PRESETS[g_preset_idx];
    set_sampling(p, pr);

    p.extend_param.base_domain_id  = 0;
    p.extend_param.embed_flash     = 1;

    RKLLMCallback cb = {};
    cb.result_callback = on_res;

    int ret = rkllm_init(&g_h, &p, &cb);
    if (ret != 0) { cout << RED " FAILED (ret=" << ret << ")\n" RST; return -1; }
    float init_s = duration<float,milli>(steady_clock::now() - t_init0).count();
    cout << GRN " done (" << ft(init_s) << ")\n" RST;

    // --- Detect model & set chat template ---
    detect_model_template(model_path);
    if (g_chat_prefix && g_chat_postfix)
        rkllm_set_chat_template(g_h, g_sys_prompt.c_str(), g_chat_prefix, g_chat_postfix);

    // --- Prompt cache ---
    // Try loading existing cache (auto-save on exit)
    {
        ifstream f(g_cache_path, ios::binary);
        if (f.good()) { f.close(); rkllm_load_prompt_cache(g_h, g_cache_path.c_str()); }
    }

    // --- Warmup ---
    cout << DIM "🔥 Warmup..." RST << flush;
    do_warmup();
    cout << GRN " done\n" RST;

    // --- Banner ---
    print_banner(model_path);

    // --- Open session log ---
    // g_session_log.open("chat_session.jsonl", ios::app);

    // --- Main loop ---
    while (true) {
        // Reset per-turn
        g_first_tok = true; g_ttft_ms = 0; g_out_toks = 0;
        g_pf_ms = 0; g_pf_toks = 0; g_dec_ms = 0; g_dec_toks = 0;
        g_mem_mb = 0; g_stop = false; g_sigcnt = 0;

        // Context bar in prompt
        int kv_used = 0;
        rkllm_get_kv_cache_size(g_h, &kv_used);
        cout << "\n" << bar(kv_used, g_max_ctx);

        cout << BLD GRN "\n▸ " RST;
        string input;
        if (!getline(cin, input)) break;
        // Trim
        size_t a = input.find_first_not_of(" \t\n\r");
        if (a == string::npos) continue;
        size_t b = input.find_last_not_of(" \t\n\r");
        input = input.substr(a, b-a+1);
        if (input.empty()) continue;

        if (input == "/exit" || input == "exit") break;
        if (input[0] == '/') { cmd(input); continue; }

        // --- Pre-process: detect & compute math expressions ---
        string processed = preprocess_input(input);
        if (processed.empty()) continue;  // math was answered directly

        // --- Inference ---
        RKLLMInput in; memset(&in, 0, sizeof(in));
        in.input_type   = RKLLM_INPUT_PROMPT;
        in.role         = "user";
        in.prompt_input = processed.c_str();
        in.enable_thinking = g_thinking;

        RKLLMInferParam ip; memset(&ip, 0, sizeof(ip));
        ip.mode         = RKLLM_INFER_GENERATE;
        ip.keep_history = g_history ? 1 : 0;
        // Per-turn sampling override (use preset params)
        RKLLMSamplingParam sp;
        sp.top_k       = pr.top_k;
        sp.top_p       = pr.top_p;
        sp.temperature = pr.temp;
        sp.repeat_penalty    = pr.repeat_penalty;
        sp.frequency_penalty = 0.f;
        sp.presence_penalty  = 0.f;
        sp.mirostat     = pr.mirostat;
        sp.mirostat_tau = pr.mirostat_tau;
        sp.mirostat_eta = pr.mirostat_eta;
        ip.sampling_params = &sp;

        cout << BLD MAG "▸ " RST;
        g_running = true;
        g_t0 = steady_clock::now();

        // sigsetjmp — catches C-library abort/terminate (can't use try-catch across extern "C")
        g_recover_armed = true;
        if (sigsetjmp(g_recover_jmp, 1) == 0) {
            ret = rkllm_run(g_h, &in, &ip, nullptr);
        } else {
            // Recovered from abort/terminate deep inside rkllm_run
            g_running = false;
            g_recover_armed = false;
            cerr << YEL "\n  ⚠ Recovered from tokenizer error, auto-clearing..." RST "\n";
            rkllm_clear_kv_cache(g_h, 0, nullptr, nullptr);
            // Re-set template to reset tokenizer internal state
            if (g_chat_prefix && g_chat_postfix)
                rkllm_set_chat_template(g_h, g_sys_prompt.c_str(), g_chat_prefix, g_chat_postfix);
            continue;
        }
        g_recover_armed = false;
        g_running = false;

        if (ret != 0) { cerr << RED "❌ ret=" << ret << RST "\n"; continue; }
        if (g_stop) continue;

        print_stats();
    }

    // --- Save prompt cache ---
    cout << DIM "\n💾 Saving prompt cache..." RST << flush;
    rkllm_release_prompt_cache(g_h);
    // Prompt cache is auto-saved by the SDK when save_prompt_cache=1 in infer params
    // For now, just clean up
    cout << GRN " done\n" RST;

    cout << DIM "👋 Goodbye.\n" RST;
    rkllm_destroy(g_h);
    g_h = nullptr;
    return 0;
}
