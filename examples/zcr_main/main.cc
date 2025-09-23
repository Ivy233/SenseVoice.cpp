//
// Created by lovemefan on 2024/7/21.
//
#include "common.h"
#include "sense-voice.h"
#include "silero-vad.h"
#include <cmath>
#include <cstdint>
#include <thread>
#include <fstream>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)// possible loss of data
#endif

// 从ifstream读取音频chunk
bool read_audio_chunk(std::ifstream &file, std::vector<float> &chunk, size_t samples_to_read) {
    std::vector<int16_t> temp_buffer(samples_to_read);
    file.read(reinterpret_cast<char*>(temp_buffer.data()), samples_to_read * sizeof(int16_t));
    
    if (file.gcount() == 0) {
        return false; // 文件结束
    }
    
    size_t actual_samples = file.gcount() / sizeof(int16_t);
    chunk.resize(actual_samples);
    
    float scale = 1.0f; // 保持与原有代码一致的缩放
    for (size_t i = 0; i < actual_samples; i++) {
        chunk[i] = static_cast<float>(temp_buffer[i]) / scale;
    }
    
    return true;
}

// command-line parameters
struct sense_voice_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t progress_step = 5;
    int32_t max_context = -1;
    int32_t n_mel = 80;
    int32_t audio_ctx = 0;
    size_t chunk_size = 50;                        // ms
    size_t max_nomute_chunks = 30000 / chunk_size;  // chunks
    size_t min_mute_chunks = 1000 / chunk_size;     // chunks
    size_t max_chunks_in_batch = 90000 / chunk_size;// chunks
    size_t max_batch = 4;

    float speech_prob_threshold = 0.1f;           // speech probability threshold
    bool debug_mode = false;
    bool no_prints = false;
    bool use_gpu = true;
    bool flash_attn = false;
    bool use_itn = false;
    bool use_prefix = true;

    std::string language = "auto";
    std::string prompt;
    std::string model = "models/ggml-base.en.bin";
    std::string openvino_encode_device = "CPU";
    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};
    std::string outfile = "";
};

static int sense_voice_has_coreml(void) {
#ifdef SENSE_VOICE_USE_COREML
    return 1;
#else
    return 0;
#endif
}

static int sense_voice_has_openvino(void) {
#ifdef SENSE_VOICE_USE_OPENVINO
    return 1;
#else
    return 0;
#endif
}

const char *sense_voice_print_system_info(void) {
    static std::string s;

    s = "";
    s += "AVX = " + std::to_string(ggml_cpu_has_avx()) + " | ";
    s += "AVX2 = " + std::to_string(ggml_cpu_has_avx2()) + " | ";
    s += "AVX512 = " + std::to_string(ggml_cpu_has_avx512()) + " | ";
    s += "FMA = " + std::to_string(ggml_cpu_has_fma()) + " | ";
    s += "NEON = " + std::to_string(ggml_cpu_has_neon()) + " | ";
    s += "ARM_FMA = " + std::to_string(ggml_cpu_has_arm_fma()) + " | ";
    s += "F16C = " + std::to_string(ggml_cpu_has_f16c()) + " | ";
    s += "FP16_VA = " + std::to_string(ggml_cpu_has_fp16_va()) + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "SSE3 = " + std::to_string(ggml_cpu_has_sse3()) + " | ";
    s += "SSSE3 = " + std::to_string(ggml_cpu_has_ssse3()) + " | ";
    s += "VSX = " + std::to_string(ggml_cpu_has_vsx()) + " | ";
    s += "COREML = " + std::to_string(sense_voice_has_coreml()) + " | ";
    s += "OPENVINO = " + std::to_string(sense_voice_has_openvino());

    return s.c_str();
}

static void sense_voice_print_usage(int /*argc*/, char **argv, const sense_voice_params &params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0.wav file1.wav ...\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n", params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n", params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n", params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -ac N,     --audio-ctx N       [%-7d] audio context size (0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n", params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -of FNAME, --output-file FNAME [%-7s] output file path (without file extension)\n", "");
    fprintf(stderr, "  -np,       --no-prints         [%-7s] do not print anything other than the results\n", params.no_prints ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect), support [`zh`, `en`, `yue`, `ja`, `ko`\n", params.language.c_str());
    fprintf(stderr, "             --use-prefix        [%-7s] use sense voice prefix\n", params.use_prefix ? "true" : "false");
    fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt (max n_text_ctx/2 tokens)\n", params.prompt.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] input WAV file path\n", "");
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n", params.openvino_encode_device.c_str());
    fprintf(stderr, "  -ng,       --no-gpu            [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,       --flash-attn        [%-7s] flash attention\n", params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -itn,      --use-itn           [%-7s] use itn\n", params.use_itn ? "true" : "false");
    fprintf(stderr, "  -fout      --outfile           [%s] output file path\n", params.outfile.c_str());
    fprintf(stderr, "             --chunk_size        [%-7lu] vad chunk size(ms)\n", params.chunk_size);
    fprintf(stderr, "  -mmc       --min-mute-chunks   [%-7lu] When consecutive chunks are identified as silence\n", params.min_mute_chunks);
    fprintf(stderr, "  -mnc       --max-nomute-chunks [%-7lu] when the first non-silent chunk is too far away\n", params.max_nomute_chunks);
    fprintf(stderr, "             --maxchunk-in-batch [%-7lu] the number of cutted audio can be processed at one time\n", params.max_chunks_in_batch);
    fprintf(stderr, "  -b         --batch             [%-7lu] the number of cutted audio can be processed at one time\n", params.max_batch);
    fprintf(stderr, "  -spt       --speech-prob-threshold [%-7.3f] speech probability threshold for VAD\n", params.speech_prob_threshold);
    fprintf(stderr, "\n");
}

struct sense_voice_print_user_data {
    const sense_voice_params *params;

    const std::vector<std::vector<float>> *pcmf32s;
    int progress_prev;
};

static char *sense_voice_param_turn_lowercase(char *in) {
    int string_len = strlen(in);
    for (int i = 0; i < string_len; i++) {
        *(in + i) = tolower((unsigned char) *(in + i));
    }
    return in;
}

static bool sense_voice_params_parse(int argc, char **argv, sense_voice_params &params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-") {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            sense_voice_print_usage(argc, argv, params);
            exit(0);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--processors") {
            params.n_processors = std::stoi(argv[++i]);
        } else if (arg == "-ot" || arg == "--offset-t") {
            params.offset_t_ms = std::stoi(argv[++i]);
        } else if (arg == "-on" || arg == "--offset-n") {
            params.offset_n = std::stoi(argv[++i]);
        } else if (arg == "-d" || arg == "--duration") {
            params.duration_ms = std::stoi(argv[++i]);
        } else if (arg == "-mc" || arg == "--max-context") {
            params.max_context = std::stoi(argv[++i]);
        } else if (arg == "-ac" || arg == "--audio-ctx") {
            params.audio_ctx = std::stoi(argv[++i]);
        } else if (arg == "-debug" || arg == "--debug-mode") {
            params.debug_mode = true;
        } else if (arg == "-of" || arg == "--output-file") {
            params.fname_out.emplace_back(argv[++i]);
        } else if (arg == "-np" || arg == "--no-prints") {
            params.no_prints = true;
        } else if (arg == "-l" || arg == "--language") {
            params.language = sense_voice_param_turn_lowercase(argv[++i]);
        } else if (arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-f" || arg == "--file") {
            params.fname_inp.emplace_back(argv[++i]);
        } else if (arg == "--use-prefix") {
            params.use_prefix = true;
        } else if (arg == "-oved" || arg == "--ov-e-device") {
            params.openvino_encode_device = argv[++i];
        } else if (arg == "-ng" || arg == "--no-gpu") {
            params.use_gpu = false;
        } else if (arg == "-fa" || arg == "--flash-attn") {
            params.flash_attn = true;
        } else if (arg == "-itn" || arg == "--use-itn") {
            params.use_itn = true;
        } else if (arg == "-mmc" || arg == "--min-mute-chunks") {
            params.min_mute_chunks = std::stoi(argv[++i]);
        } else if (arg == "-mnc" || arg == "--max-nomute-chunks") {
            params.max_nomute_chunks = std::stoi(argv[++i]);
        } else if (arg == "--maxchunk-in-batch") {
            params.max_chunks_in_batch = std::stoi(argv[++i]);
        } else if (arg == "-b" || arg == "--batch") {
            params.max_batch = std::stoi(argv[++i]);
        } else if (arg == "--chunk_size") {
            params.chunk_size = std::stoi(argv[++i]);
        } else if (arg == "--outfile" || arg == "-fout") {
            params.outfile = argv[++i];
        } else if (arg == "-spt" || arg == "--speech-prob-threshold") {
            params.speech_prob_threshold = std::stof(argv[++i]);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sense_voice_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

static bool is_file_exist(const char *fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

// 函数声明
void sense_voice_process_stream_from_file(struct sense_voice_context *ctx, const sense_voice_params &params, 
                                         std::ifstream &file, const WaveHeader &header);
void sense_voice_process_batch(struct sense_voice_context *ctx, const sense_voice_params &params,
                               std::vector<sense_voice_segment> &batch);
bool check_and_process_batch_if_full(struct sense_voice_context *ctx, const sense_voice_params &params,
                                     std::vector<sense_voice_segment> &current_batch, size_t &current_batch_size,
                                     size_t new_segment_size, size_t batch_samples);

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor *t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor *t, bool ask, void *user_data) {
    auto *cb_data = (callback_data *) user_data;

    const struct ggml_tensor *src0 = t->src[0];
    const struct ggml_tensor *src1 = t->src[1];

    if (ask) {
        return true;// Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
           t->name, ggml_type_name(t->type), ggml_op_desc(t),
           src0->name, ggml_ne_string(src0).c_str(),
           src1 ? src1_str : "",
           ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t *data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        // ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
    }

    return true;
}

void sense_voice_free(struct sense_voice_context *ctx) {
    if (ctx) {
        ggml_free(ctx->model.ctx);
        ggml_backend_buffer_free(ctx->model.buffer);

        // 释放VAD相关资源 - 添加空指针检查
        if (ctx->state) {
            if (ctx->state->vad_ctx) {
                ggml_free(ctx->state->vad_ctx);
                ctx->state->vad_ctx = nullptr;
            }
            if (ctx->state->vad_lstm_hidden_state_buffer) {
                ggml_backend_buffer_free(ctx->state->vad_lstm_hidden_state_buffer);
                ctx->state->vad_lstm_hidden_state_buffer = nullptr;
            }
            if (ctx->state->vad_lstm_context_buffer) {
                ggml_backend_buffer_free(ctx->state->vad_lstm_context_buffer);
                ctx->state->vad_lstm_context_buffer = nullptr;
            }
        }

        sense_voice_free_state(ctx->state);
        delete ctx->model.model->encoder;
        delete ctx->model.model;
        delete ctx;
    }
}

// 流式音频处理：从ifstream逐块读取并处理
void sense_voice_process_stream_from_file(struct sense_voice_context *ctx, const sense_voice_params &params, 
                                         std::ifstream &file, const WaveHeader &header) {
    const int n_sample_step = params.chunk_size * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const int keep_nomute_step = params.chunk_size * params.min_mute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const int max_nomute_step = params.chunk_size * params.max_nomute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const size_t batch_samples = params.max_chunks_in_batch * params.chunk_size * 1e-3 * SENSE_VOICE_SAMPLE_RATE;

    std::vector<sense_voice_segment> current_batch;
    size_t current_batch_size = 0;

    int L_nomute = -1, L_mute = -1, R_mute = -1;
    
    // 流式读取缓冲区
    std::vector<float> audio_buffer;
    std::vector<float> chunk_data;
    const size_t chunk_samples = n_sample_step;
    int processed_samples = 0;

    // 逐块读取音频数据
    while (read_audio_chunk(file, chunk_data, chunk_samples)) {
        // 将新数据追加到缓冲区
        audio_buffer.insert(audio_buffer.end(), chunk_data.begin(), chunk_data.end());
        
        // 处理缓冲区中的完整chunks
        while (audio_buffer.size() >= processed_samples + n_sample_step) {
            int i = processed_samples;
            int R_this_chunk = std::min(i + n_sample_step, (int)audio_buffer.size());
            bool isnomute = false;

            // VAD检测
            int actual_chunk_size = R_this_chunk - i;
            int vad_chunk_size = std::max(640, actual_chunk_size);
            std::vector<float> vad_chunk(vad_chunk_size, 0);

            for (int j = 0; j < actual_chunk_size && i + j < (int)audio_buffer.size(); j++) {
                vad_chunk[j] = audio_buffer[i + j] / 32768.0f;
            }

            if (actual_chunk_size < 640) {
                float last_sample = (actual_chunk_size > 0) ? vad_chunk[actual_chunk_size - 1] : 0.0f;
                for (int j = actual_chunk_size; j < 640; j++) {
                    vad_chunk[j] = last_sample;
                }
            }

            float speech_prob = 0;
            if (silero_vad_encode_internal(*ctx, *ctx->state, vad_chunk, params.n_threads, speech_prob)) {
                isnomute = (speech_prob >= params.speech_prob_threshold);
            } else {
                isnomute = vad_energy_zcr<float>(audio_buffer.begin() + i, R_this_chunk - i, SENSE_VOICE_SAMPLE_RATE);
            }

            // 音频分段逻辑（与原来的逻辑相同）
            if (L_nomute >= 0 && R_this_chunk - L_nomute >= max_nomute_step) {
                int R_nomute = L_mute >= 0 && L_mute >= L_nomute ? L_mute : R_this_chunk;
                sense_voice_segment segment;
                segment.t0 = L_nomute;
                segment.t1 = R_nomute;
                segment.samples = std::vector<float>(audio_buffer.begin() + L_nomute, audio_buffer.begin() + R_nomute);

                size_t segment_size = segment.samples.size();
                check_and_process_batch_if_full(ctx, params, current_batch, current_batch_size, 
                                              segment_size, batch_samples);
                
                current_batch.push_back(segment);
                current_batch_size += segment_size;

                if (!isnomute) L_nomute = -1;
                else if (R_mute >= 0 && L_mute >= L_nomute) L_nomute = R_mute;
                else L_nomute = i;
                L_mute = R_mute = -1;
                
                processed_samples = R_this_chunk;
                continue;
            }

            if (isnomute) {
                if (L_nomute < 0) L_nomute = i;
            } else {
                if (R_mute != i) L_mute = i;
                R_mute = R_this_chunk;
                if (L_mute >= L_nomute && L_nomute >= 0 && R_this_chunk - L_mute >= keep_nomute_step) {
                    sense_voice_segment segment;
                    segment.t0 = L_nomute;
                    segment.t1 = L_mute;
                    segment.samples = std::vector<float>(audio_buffer.begin() + L_nomute, audio_buffer.begin() + L_mute);

                    size_t segment_size = segment.samples.size();
                    check_and_process_batch_if_full(ctx, params, current_batch, current_batch_size, 
                                                  segment_size, batch_samples);
                    
                    current_batch.push_back(segment);
                    current_batch_size += segment_size;

                    if (!isnomute) L_nomute = -1;
                    else if (R_mute >= 0) L_nomute = R_mute;
                    else L_nomute = i;
                    L_mute = R_mute = -1;
                }
            }
            
            processed_samples = R_this_chunk;
        }
        
        // 定期清理已处理的缓冲区数据，防止内存无限增长
        if (processed_samples > 0 && audio_buffer.size() > 2 * max_nomute_step) {
            std::vector<float> temp_buffer(audio_buffer.begin() + processed_samples, audio_buffer.end());
            audio_buffer = std::move(temp_buffer);
            
            // 调整位置索引
            L_nomute -= processed_samples;
            L_mute -= processed_samples;
            R_mute -= processed_samples;
            if (L_nomute < 0 && L_nomute != -1) L_nomute = -1;
            if (L_mute < 0 && L_mute != -1) L_mute = -1;
            if (R_mute < 0 && R_mute != -1) R_mute = -1;
            
            processed_samples = 0;
        }
    }

    // 处理最后一段
    if (L_nomute >= 0 && L_nomute < (int)audio_buffer.size()) {
        sense_voice_segment segment;
        segment.t0 = L_nomute;
        segment.t1 = audio_buffer.size();
        segment.samples = std::vector<float>(audio_buffer.begin() + L_nomute, audio_buffer.end());

        size_t segment_size = segment.samples.size();
        check_and_process_batch_if_full(ctx, params, current_batch, current_batch_size, 
                                      segment_size, batch_samples);
        
        current_batch.push_back(segment);
    }

    // 处理最后的batch
    if (!current_batch.empty()) {
        sense_voice_process_batch(ctx, params, current_batch);
    }
}

// 处理一个batch并清理计算图缓冲区
void sense_voice_process_batch(struct sense_voice_context *ctx, const sense_voice_params &params,
                               std::vector<sense_voice_segment> &batch) {
    // 清理之前的结果
    ctx->state->result_all.clear();
    ctx->state->segmentIDs.clear();

    // 将batch中的segment添加到result_all中
    for (size_t i = 0; i < batch.size(); i++) {
        ctx->state->result_all.push_back(batch[i]);
        ctx->state->segmentIDs.push_back(i);
    }

    // 处理batch
    sense_voice_full_params wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
    wparams.language = params.language.c_str();
    wparams.n_threads = params.n_threads;
    wparams.debug_mode = params.debug_mode;

    sense_voice_batch_full(ctx, wparams);
    sense_voice_batch_print_output(ctx, params.use_prefix, params.use_itn);

    // 清理处理后的结果以释放内存
    ctx->state->result_all.clear();
    ctx->state->segmentIDs.clear();
}

// 检查batch是否满载，如果满载则处理并清空
bool check_and_process_batch_if_full(struct sense_voice_context *ctx, const sense_voice_params &params,
                                     std::vector<sense_voice_segment> &current_batch, size_t &current_batch_size,
                                     size_t new_segment_size, size_t batch_samples) {
    if (!current_batch.empty() && 
        (current_batch_size + new_segment_size > batch_samples || 
         current_batch.size() >= params.max_batch)) {
        
        // 处理当前batch
        sense_voice_process_batch(ctx, params, current_batch);
        
        // 清空batch准备下一轮
        current_batch.clear();
        current_batch_size = 0;
        
        return true; // 表示已处理了一个batch
    }
    return false; // 表示未处理batch
}

int main(int argc, char **argv) {
    sense_voice_params params;

    if (!sense_voice_params_parse(argc, argv, params)) {
        sense_voice_print_usage(argc, argv, params);
        return 1;
    }

    // remove non-existent files
    for (auto it = params.fname_inp.begin(); it != params.fname_inp.end();) {
        const auto fname_inp = it->c_str();

        if (*it != "-" && !is_file_exist(fname_inp)) {
            fprintf(stderr, "error: input file not found '%s'\n", fname_inp);
            it = params.fname_inp.erase(it);
            continue;
        }

        it++;
    }

    if (params.fname_inp.empty()) {
        fprintf(stderr, "error: no input files specified\n");
        sense_voice_print_usage(argc, argv, params);
        return 2;
    }

    if (params.language != "auto" && sense_voice_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        sense_voice_print_usage(argc, argv, params);
        exit(0);
    }

    if (!params.outfile.empty()) {
        freopen(params.outfile.c_str(), "w", stdout);
        params.use_prefix = false;
    }

    // sense-voice init

    struct sense_voice_context_params cparams = sense_voice_context_default_params();

    callback_data cb_data;

    cparams.cb_eval = ggml_debug;
    cparams.cb_eval_user_data = &cb_data;

    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    cparams.use_itn = params.use_itn;

    struct sense_voice_context *ctx = sense_voice_small_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize sense voice context\n");
        return 3;
    }

    ctx->language_id = sense_voice_lang_id(params.language.c_str());

    // 初始化silero-vad状态
    const int VAD_LSTM_STATE_MEMORY_SIZE = 2048;
    const int VAD_LSTM_STATE_DIM = 128;

    ctx->state->vad_ctx = ggml_init({VAD_LSTM_STATE_MEMORY_SIZE, nullptr, true});
    ctx->state->vad_lstm_context = ggml_new_tensor_1d(ctx->state->vad_ctx, GGML_TYPE_F32, VAD_LSTM_STATE_DIM);
    ctx->state->vad_lstm_hidden_state = ggml_new_tensor_1d(ctx->state->vad_ctx, GGML_TYPE_F32, VAD_LSTM_STATE_DIM);

    ctx->state->vad_lstm_context_buffer = ggml_backend_alloc_buffer(ctx->state->backends[0],
                                                                    ggml_nbytes(ctx->state->vad_lstm_context)
                                                                            + ggml_backend_get_alignment(ctx->state->backends[0]));
    ctx->state->vad_lstm_hidden_state_buffer = ggml_backend_alloc_buffer(ctx->state->backends[0],
                                                                         ggml_nbytes(ctx->state->vad_lstm_hidden_state)
                                                                                 + ggml_backend_get_alignment(ctx->state->backends[0]));
    auto context_alloc = ggml_tallocr_new(ctx->state->vad_lstm_context_buffer);
    ggml_tallocr_alloc(&context_alloc, ctx->state->vad_lstm_context);

    auto state_alloc = ggml_tallocr_new(ctx->state->vad_lstm_hidden_state_buffer);
    ggml_tallocr_alloc(&state_alloc, ctx->state->vad_lstm_hidden_state);

    ggml_set_zero(ctx->state->vad_lstm_context);
    ggml_set_zero(ctx->state->vad_lstm_hidden_state);

    for (int f = 0; f < (int) params.fname_inp.size(); ++f) {
        const auto fname_inp = params.fname_inp[f];
        const auto fname_out = f < (int) params.fname_out.size() && !params.fname_out[f].empty() ? params.fname_out[f] : params.fname_inp[f];

        // 使用ifstream进行流式音频读取
        std::ifstream file(fname_inp.c_str(), std::ios::binary);
        if (!file) {
            fprintf(stderr, "error: failed to open audio file '%s'\n", fname_inp.c_str());
            continue;
        }

        // 读取WAV头
        WaveHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file || !header.Validate()) {
            fprintf(stderr, "error: invalid WAV file format '%s'\n", fname_inp.c_str());
            continue;
        }

        header.SeekToDataChunk(file);
        if (!file) {
            fprintf(stderr, "error: failed to find data chunk in '%s'\n", fname_inp.c_str());
            continue;
        }

        int sample_rate = header.sample_rate;
        size_t total_samples = header.subchunk2_size / 2; // 16-bit samples

        if (!params.no_prints) {
            // print system information
            fprintf(stderr, "\n");
            fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                    params.n_threads * params.n_processors, std::thread::hardware_concurrency(), sense_voice_print_system_info());

            // print some info about the processing
            fprintf(stderr, "\n");
            fprintf(stderr, "%s: processing audio stream (%zu samples, %.5f sec) , %d threads, %d processors, lang = %s...\n",
                    __func__, total_samples, float(total_samples) / sample_rate,
                    params.n_threads, params.n_processors,
                    params.language.c_str());
            ctx->state->duration = float(total_samples) / sample_rate;
            fprintf(stderr, "\n");
        }

        {
            // 使用流式处理音频
            sense_voice_process_stream_from_file(ctx, params, file, header);
        }
        
        file.close();
    }
    sense_voice_free(ctx);
    return 0;
}