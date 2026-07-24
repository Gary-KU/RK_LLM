# RKLLM SDK API 完整参考手册

> 基于 RKLLM SDK v1.3.0 · 适用于 RK3588 / RK3576 / RK3562 / RV1126B
>
> 源码：`sdk/rkllm-runtime/Android/librkllm_api/include/rkllm.h`

---

## 目录

1. [数据结构](#1-数据结构)
   - [LLMHandle](#llmhandle)
   - [枚举类型](#枚举类型)
   - [RKLLMParam — 模型参数](#rkllmparam--模型参数)
   - [RKLLMExtendParam — 扩展参数](#rkllmextendparam--扩展参数)
   - [RKLLMInput — 推理输入](#rkllminput--推理输入)
   - [RKLLMInferParam — 推理控制](#rkllminferparam--推理控制)
   - [RKLLMSamplingParam — 采样参数](#rkllmsamplingparam--采样参数)
   - [RKLLMResult — 推理结果](#rkllmresult--推理结果)
   - [RKLLMPerfStat — 性能统计](#rkllmperfstat--性能统计)
   - [RKLLMCallback — 回调集合](#rkllmcallback--回调集合)
   - [其他结构体](#其他结构体)
2. [API 函数](#2-api-函数)
   - [生命周期](#生命周期)
   - [推理](#推理)
   - [对话模板](#对话模板)
   - [KV Cache](#kv-cache)
   - [LoRA](#lora)
   - [跨注意力](#跨注意力)
3. [CPU 掩码宏](#3-cpu-掩码宏)
4. [使用示例](#4-使用示例)

---

## 1. 数据结构

### LLMHandle

```c
typedef void* LLMHandle;
```

不透明句柄，由 `rkllm_init` 创建，所有 API 调用都传递此句柄。

---

### 枚举类型

#### LLMCallState — 回调状态

```c
typedef enum {
    RKLLM_RUN_NORMAL  = 0,  // 正常生成中（每个 token 触发一次）
    RKLLM_RUN_WAITING = 1,  // 等待 UTF-8 完整字符
    RKLLM_RUN_FINISH  = 2,  // 推理完成
    RKLLM_RUN_ERROR   = 3,  // 推理出错
} LLMCallState;
```

#### RKLLMInputType — 输入类型

```c
typedef enum {
    RKLLM_INPUT_PROMPT      = 0,  // 文本 prompt
    RKLLM_INPUT_TOKEN       = 1,  // token ID 序列
    RKLLM_INPUT_EMBED       = 2,  // 嵌入向量
    RKLLM_INPUT_MULTIMODAL  = 3,  // 多模态（文本+图片/视频）
} RKLLMInputType;
```

#### RKLLMInferMode — 推理模式

```c
typedef enum {
    RKLLM_INFER_GENERATE              = 0,  // 文本生成（最常用）
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1,  // 获取最后一层隐状态
    RKLLM_INFER_GET_LOGITS            = 2,  // 获取 logits
} RKLLMInferMode;
```

---

### RKLLMParam — 模型参数

`rkllm_init` 的核心参数结构体，控制模型加载和默认采样策略。

```c
typedef struct {
    const char* model_path;         // [必填] 模型文件路径 (.rkllm)
    int32_t max_context_len;        // [必填] 上下文窗口最大 token 数
    int32_t max_new_tokens;         // [必填] 单次最大生成 token 数

    // ── 采样参数（初始化时设置，推理时可通过 RKLLMSamplingParam 覆盖）──
    int32_t top_k;                  // Top-K 采样 (1 = 贪婪)
    float   top_p;                  // Top-P / nucleus 采样 (0.0-1.0)
    float   temperature;            // 温度，越高越随机 (0.1-2.0)
    float   repeat_penalty;         // 重复惩罚 (1.0 = 不惩罚)
    float   frequency_penalty;      // 频率惩罚
    float   presence_penalty;       // 存在惩罚
    int32_t mirostat;               // Mirostat 采样策略 (0 = 禁用, 2 = v2)
    float   mirostat_tau;           // Mirostat τ 参数
    float   mirostat_eta;           // Mirostat η 参数

    int32_t n_keep;                 // KV 滑动时保留开头的 token 数
    bool    skip_special_token;     // 是否跳过特殊 token 输出
    bool    ignore_eos_token;       // 是否忽略 EOS token
    bool    is_async;               // 是否异步推理模式

    RKLLMExtendParam extend_param;  // 扩展参数（硬件相关）
} RKLLMParam;
```

**工厂函数**：

```c
RKLLMParam rkllm_createDefaultParam();  // 创建带默认值的 RKLLMParam
```

---

### RKLLMExtendParam — 扩展参数

```c
typedef struct {
    int32_t  base_domain_id;     // NPU domain ID，通常设 0
    int8_t   embed_flash;        // 从 flash 查询词嵌入向量 (1=是)
    int8_t   enabled_cpus_num;   // 启用的 CPU 核数（与 enabled_cpus_mask 互斥）
    uint32_t enabled_cpus_mask;  // CPU 核掩码（使用 CPU0~CPU7 宏组合）
    uint8_t  n_batch;            // Prefill 并行 token 数 (默认 1)
    int8_t   use_cross_attn;     // 启用跨注意力 (1=是)
    uint8_t  reserved[104];      // 保留
} RKLLMExtendParam;
```

> **注意**：`enabled_cpus_num` 和 `enabled_cpus_mask` 互斥，设置一个即可。
> RK3588 上建议不手动设，使用 SDK 默认值。

---

### RKLLMInput — 推理输入

每次调用 `rkllm_run` 时传入的输入数据。

```c
typedef struct {
    const char* role;             // 消息角色: "user" (用户) 或 "tool" (工具返回)
    bool enable_thinking;         // Qwen3 思考模式开关
    RKLLMInputType input_type;    // 输入类型

    union {
        const char* prompt_input;                // 文本 prompt (RKLLM_INPUT_PROMPT)
        RKLLMEmbedInput embed_input;             // 嵌入 (RKLLM_INPUT_EMBED)
        RKLLMTokenInput token_input;             // Token (RKLLM_INPUT_TOKEN)
        RKLLMMultiModalInput multimodal_input;   // 多模态 (RKLLM_INPUT_MULTIMODAL)
    };
} RKLLMInput;
```

**使用示例**：

```c
RKLLMInput in = {};
in.input_type   = RKLLM_INPUT_PROMPT;
in.role         = "user";
in.prompt_input = "你好，请介绍一下自己";
```

---

### RKLLMInferParam — 推理控制

```c
typedef struct {
    RKLLMInferMode mode;                        // 推理模式
    RKLLMLoraParam* lora_params;                // LoRA 适配器参数（可为 NULL）
    RKLLMPromptCacheParam* prompt_cache_params; // Prompt Cache 参数（可为 NULL）
    RKLLMSamplingParam* sampling_params;        // 采样参数覆盖（可为 NULL）
    int keep_history;                           // 是否保留会话历史 (1=是, 0=否)
    int32_t max_new_tokens;                     // 覆盖最大生成 token 数（0=使用 init 值）
} RKLLMInferParam;
```

**使用示例**：

```c
RKLLMInferParam ip = {};
ip.mode         = RKLLM_INFER_GENERATE;
ip.keep_history = 1;  // 保留对话历史
```

---

### RKLLMSamplingParam — 采样参数

可在每轮推理时覆盖初始化时的采样设置。

```c
typedef struct {
    int32_t top_k;
    float   top_p;
    float   temperature;
    float   repeat_penalty;
    float   frequency_penalty;
    float   presence_penalty;
    int32_t mirostat;
    float   mirostat_tau;
    float   mirostat_eta;
} RKLLMSamplingParam;
```

**使用示例**（实时切换采样策略）：

```c
RKLLMSamplingParam sp = {};
sp.top_k = 1;          // 贪婪解码（代码/数学场景）
sp.temperature = 0.1f;
ip.sampling_params = &sp;
rkllm_run(handle, &input, &ip, NULL);
```

---

### RKLLMResult — 推理结果

在回调函数中接收，包含生成的文本和性能数据。

```c
typedef struct {
    const char* text;                              // 本次回调生成的文本片段
    int32_t token_id;                              // 生成 token 的 ID
    RKLLMResultLastHiddenLayer last_hidden_layer;  // 隐状态（特定模式）
    RKLLMResultLogits logits;                      // logits（特定模式）
    RKLLMPerfStat perf;                           // 性能统计
} RKLLMResult;
```

**Last Hidden Layer 子结构**：

```c
typedef struct {
    const float* hidden_states;  // 隐状态数组 (大小: num_tokens × embd_size)
    int embd_size;               // 嵌入维度
    int num_tokens;              // token 数量
} RKLLMResultLastHiddenLayer;
```

**Logits 子结构**：

```c
typedef struct {
    const float* logits;    // logits 数组 (大小: num_tokens × vocab_size)
    int vocab_size;          // 词表大小
    int num_tokens;          // token 数量
} RKLLMResultLogits;
```

---

### RKLLMPerfStat — 性能统计

推理完成后可获取的详细性能数据。

```c
typedef struct {
    float prefill_time_ms;     // Prefill 阶段总耗时 (ms)
    int   prefill_tokens;      // Prefill 处理的 token 数
    float generate_time_ms;    // Decode 阶段总耗时 (ms)
    int   generate_tokens;     // Decode 生成的 token 数
    float memory_usage_mb;     // 峰值内存 (VmHWM, MB)
} RKLLMPerfStat;
```

**在回调中获取**：

```c
int callback(RKLLMResult* r, void*, LLMCallState st) {
    if (st == RKLLM_RUN_FINISH) {
        printf("TTFT: %.0fms\n", r->perf.prefill_time_ms);
        printf("Decode: %.1f tok/s\n",
            r->perf.generate_tokens / (r->perf.generate_time_ms / 1000.0));
        printf("Memory: %.0f MB\n", r->perf.memory_usage_mb);
    }
    return 0;
}
```

---

### RKLLMCallback — 回调集合

```c
typedef struct {
    LLMResultCallback   result_callback;    // 推理结果回调（必须）
    void*               result_userdata;    // 结果回调的用户数据
    LLMTokenizerCallback tokenizer_callback; // 自定义分词器（可选）
    void*                tokenizer_userdata;
    LLMGetEmbedCallback  embed_callback;     // 自定义嵌入层（可选）
    void*                embed_userdata;
} RKLLMCallback;
```

**结果回调签名**：

```c
typedef int (*LLMResultCallback)(RKLLMResult* result, void* userdata, LLMCallState state);

// 返回值:
//   0 — 继续推理
//   1 — 暂停推理（用于内容拦截/修改）
//   2 — 释放输出缓冲区（用于获取隐状态/logits 后及时释放内存）
```

---

### 其他结构体

#### LoRA 适配器

```c
typedef struct {
    const char* lora_adapter_path;  // LoRA 文件路径
    const char* lora_adapter_name;  // LoRA 名称
    float scale;                    // 缩放系数
} RKLLMLoraAdapter;

typedef struct {
    const char* lora_adapter_name;  // 推理时指定使用哪个 LoRA
} RKLLMLoraParam;
```

#### Prompt Cache

```c
typedef struct {
    int save_prompt_cache;          // 是否保存 (0/1)
    const char* prompt_cache_path;   // 缓存文件路径
} RKLLMPromptCacheParam;
```

#### 多模态输入

```c
typedef struct {
    char* prompt;
    struct {
        float* image_embed;        // 图片嵌入 (n_image × n_tokens × dim × float32)
        size_t n_image_tokens;
        size_t n_image;
        const char* image_start;   // 图片起始标记（如 "<img>"）
        const char* image_end;     // 图片结束标记（如 "</img>"）
        const char* image_content; // 图片内容占位
        size_t image_width, image_height;
    } image;
    struct {
        float* video_embed;
        size_t n_frame_tokens, n_frame_per_video, n_video;
        const char* video_start, *video_end, *video_content;
        size_t frame_width, frame_height;
    } video;
} RKLLMMultiModalInput;
```

#### 跨注意力（编码器-解码器场景）

```c
typedef struct {
    float* encoder_k_cache;    // 编码器 K cache
    float* encoder_v_cache;    // 编码器 V cache
    float* encoder_mask;       // 注意力掩码
    int32_t* encoder_pos;      // 编码器位置
    int num_tokens;
} RKLLMCrossAttnParam;
```

---

## 2. API 函数

### 生命周期

#### rkllm_createDefaultParam

```c
RKLLMParam rkllm_createDefaultParam();
```

创建带默认值的 `RKLLMParam`。建议始终从该函数获取，再覆盖需要的字段。

#### rkllm_init

```c
int rkllm_init(LLMHandle* handle, RKLLMParam* param, RKLLMCallback* callback);
```

初始化 LLM 实例。加载模型文件、分配 NPU 资源、配置回调。

- **参数**：
  - `handle`: 输出，句柄指针
  - `param`: 模型参数
  - `callback`: 回调结构体（至少设 `result_callback`）
- **返回值**：0 成功，非 0 失败

#### rkllm_destroy

```c
int rkllm_destroy(LLMHandle handle);
```

销毁 LLM 实例，释放所有资源。

---

### 推理

#### rkllm_run（同步）

```c
int rkllm_run(LLMHandle handle, RKLLMInput* rkllm_input,
              RKLLMInferParam* rkllm_infer_params, void* userdata);
```

执行一次同步推理。函数会阻塞直到推理完成（或出错）。

- **userdata**: 非 NULL 时优先作为回调的 userdata，否则使用 `RKLLMCallback.result_userdata`

#### rkllm_run_async（异步）

```c
int rkllm_run_async(LLMHandle handle, RKLLMInput* rkllm_input,
                    RKLLMInferParam* rkllm_infer_params, void* userdata);
```

执行异步推理。调用后立即返回，推理在后台执行，结果通过回调通知。

#### rkllm_abort

```c
int rkllm_abort(LLMHandle handle);
```

中止正在进行的推理任务。通常在 Ctrl+C 信号处理中调用。

#### rkllm_is_running

```c
int rkllm_is_running(LLMHandle handle);
```

查询是否有推理任务正在运行。返回值 1 = 正在运行。

---

### 对话模板

#### rkllm_set_chat_template

```c
int rkllm_set_chat_template(LLMHandle handle,
    const char* system_prompt,   // 系统提示词
    const char* prompt_prefix,   // 用户消息前缀
    const char* prompt_postfix); // 用户消息后缀（接着生成 assistant 回复）
```

设置或修改对话模板。

**Qwen3 示例**：

```c
rkllm_set_chat_template(handle,
    "你是一个乐于助人的AI助手。",
    "<|im_start|>user\n",
    "<|im_end|>\n<|im_start|>assistant\n");
```

**LLaMA 示例**：

```c
rkllm_set_chat_template(handle,
    "You are a helpful assistant.",
    "<s>[INST] ",
    " [/INST]");
```

> **注意**：prefix/postfix 不能为空字符串，否则中文输入可能触发 tokenizer 崩溃。
> 建议按模型类型自动匹配（见 [2.10 模型模板自动匹配](PERFORMANCE.md#210-模型模板自动匹配)）。

#### rkllm_set_function_tools

```c
int rkllm_set_function_tools(LLMHandle handle,
    const char* system_prompt,     // 带工具使用说明的系统提示词
    const char* tools,             // JSON 格式的工具定义
    const char* tool_response_str); // 工具返回标记（如 "<tool_response>"）
```

配置函数调用（tool calling）能力。

**tools JSON 格式（OpenAI 兼容）**：

```json
[{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "计算数学表达式",
    "parameters": {
      "type": "object",
      "properties": {
        "expr": {"type": "string", "description": "表达式"}
      },
      "required": ["expr"]
    }
  }
}]
```

> **注意**：基础 instruct 模型（如 Qwen3-4B-Instruct）未专门为 function calling 微调，
> 建议使用预处理拦截方式处理数学计算（见 [2.12 数学预处理](PERFORMANCE.md#212-数学预处理--行业标准拦截图)）。

---

### KV Cache

#### rkllm_clear_kv_cache

```c
int rkllm_clear_kv_cache(LLMHandle handle,
    int keep_system_prompt,   // 1=保留系统提示词, 0=全部清除
    int* start_pos,           // 清除范围起始（NULL=全部）
    int* end_pos);            // 清除范围结束（NULL=全部）
```

清除 KV Cache，用于重置对话上下文。

- `keep_system_prompt = 1`：只清除对话历史，保留系统提示词的 KV
- `start_pos/end_pos` 非 NULL 时按指定范围清除

**使用场景**：

```c
// 清空整个对话
rkllm_clear_kv_cache(handle, 0, NULL, NULL);

// 保留系统提示词，只清除用户对话
rkllm_clear_kv_cache(handle, 1, NULL, NULL);
```

#### rkllm_get_kv_cache_size

```c
int rkllm_get_kv_cache_size(LLMHandle handle, int* cache_sizes);
```

获取当前 KV Cache 中的 token 数量。用于监控上下文窗口使用率。

**使用示例**：

```c
int used = 0;
rkllm_get_kv_cache_size(handle, &used);
printf("[%d/%d]\n", used, max_context_len);
```

#### rkllm_load_prompt_cache

```c
int rkllm_load_prompt_cache(LLMHandle handle, const char* prompt_cache_path);
```

从文件加载 Prompt Cache（系统提示词的预计算 KV Cache），跳过系统提示词的重复 prefill。

#### rkllm_release_prompt_cache

```c
int rkllm_release_prompt_cache(LLMHandle handle);
```

释放 Prompt Cache 占用的内存。

**Prompt Cache 完整流程**：

```c
// 首次运行
RKLLMPromptCacheParam pc = {};
pc.save_prompt_cache = 1;
pc.prompt_cache_path = "./cache.bin";
ip.prompt_cache_params = &pc;
rkllm_run(handle, &in, &ip, NULL);  // 首次推理时保存

// 后续运行
rkllm_load_prompt_cache(handle, "./cache.bin");  // 加载，跳过系统提示词 prefill
// ... 正常推理 ...
rkllm_release_prompt_cache(handle);  // 退出时释放
```

---

### LoRA

#### rkllm_load_lora

```c
int rkllm_load_lora(LLMHandle handle, RKLLMLoraAdapter* lora_adapter);
```

加载 LoRA 适配器。可多次调用加载多个 LoRA，推理时通过 `RKLLMLoraParam` 指定使用的 LoRA。

```c
RKLLMLoraAdapter lora = {};
lora.lora_adapter_path = "my_lora.rkllm";
lora.lora_adapter_name = "my_adapter";
lora.scale = 1.0;
rkllm_load_lora(handle, &lora);

// 推理时指定
RKLLMLoraParam lp = { .lora_adapter_name = "my_adapter" };
ip.lora_params = &lp;
```

---

### 跨注意力

#### rkllm_set_cross_attn_params

```c
int rkllm_set_cross_attn_params(LLMHandle handle, RKLLMCrossAttnParam* cross_attn_params);
```

设置编码器-解码器跨注意力参数，用于多模态（图片/视频理解）等场景。

---

## 3. CPU 掩码宏

```c
#define CPU0 (1 << 0)  // 0x01  — A55 小核 0
#define CPU1 (1 << 1)  // 0x02  — A55 小核 1
#define CPU2 (1 << 2)  // 0x04  — A55 小核 2
#define CPU3 (1 << 3)  // 0x08  — A55 小核 3
#define CPU4 (1 << 4)  // 0x10  — A76 大核 0
#define CPU5 (1 << 5)  // 0x20  — A76 大核 1
#define CPU6 (1 << 6)  // 0x40  — A76 大核 2
#define CPU7 (1 << 7)  // 0x80  — A76 大核 3
```

**RK3588 典型配置**：

```c
// 只用大核 (A76 x4)
param.extend_param.enabled_cpus_mask = CPU4 | CPU5 | CPU6 | CPU7;

// 全 8 核
param.extend_param.enabled_cpus_mask = 0xFF;
```

> **注意**：`enabled_cpus_mask` 与 `enabled_cpus_num` 互斥。不设置时 SDK 使用默认值。

---

## 4. 使用示例

### 最小可用程序

```c
#include "rkllm.h"
#include <stdio.h>

LLMHandle g_handle;

int callback(RKLLMResult* r, void*, LLMCallState st) {
    if (st == RKLLM_RUN_NORMAL && r->text) printf("%s", r->text);
    if (st == RKLLM_RUN_FINISH) printf("\n");
    return 0;
}

int main() {
    // 1. 初始化
    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path      = "model.rkllm";
    p.max_context_len = 4096;
    p.max_new_tokens  = 2048;

    RKLLMCallback cb = { .result_callback = callback };
    rkllm_init(&g_handle, &p, &cb);

    // 2. 设置系统提示词
    rkllm_set_chat_template(g_handle, "You are helpful.",
        "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");

    // 3. 推理循环
    while (1) {
        char input[1024];
        printf("user: ");
        fgets(input, sizeof(input), stdin);

        RKLLMInput in = {};
        in.input_type   = RKLLM_INPUT_PROMPT;
        in.role         = "user";
        in.prompt_input = input;

        RKLLMInferParam ip = {};
        ip.mode         = RKLLM_INFER_GENERATE;
        ip.keep_history = 1;

        printf("robot: ");
        rkllm_run(g_handle, &in, &ip, NULL);
    }

    // 4. 清理
    rkllm_destroy(g_handle);
}
```

### 完整工程参考

> 生产级实现见仓库中的 **`src/rkchat.cpp`**（530 行），包含：
> - 流式输出 + 性能统计
> - 多轮对话 + KV Cache 监控
> - 数学预处理拦截
> - 崩溃恢复（sigsetjmp）
> - 模型模板自动匹配
> - 4 档采样预设

---

> **相关文档**：
> - [PERFORMANCE.md](PERFORMANCE.md) — 性能优化教程
> - [GUIDE.md](../GUIDE.md) — 详细使用手册
> - [README.md](../README.md) — 项目概览
