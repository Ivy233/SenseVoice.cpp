#include "common-sdl.h"
#include "common.h"
#include "sense-voice.h"
#include "silero-vad.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <cstring>

// WAV 文件头结构
struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // 单声道
    uint32_t sample_rate = SENSE_VOICE_SAMPLE_RATE;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;

    WAVHeader() {
        byte_rate = sample_rate * num_channels * bits_per_sample / 8;
        block_align = num_channels * bits_per_sample / 8;
        file_size = 0; // 将在写入时更新
        data_size = 0; // 将在写入时更新
    }
};

// 将 float 音频数据转换为 16 位 PCM
void float_to_pcm16(const std::vector<float>& float_audio, std::vector<int16_t>& pcm16_audio) {
    pcm16_audio.resize(float_audio.size());
    for (size_t i = 0; i < float_audio.size(); ++i) {
        // 限制在 [-1.0, 1.0] 范围内，然后转换为 16 位
        float sample = std::max(-1.0f, std::min(1.0f, float_audio[i]));
        pcm16_audio[i] = static_cast<int16_t>(sample * 32767.0f);
    }
}

struct sense_voice_stream_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t capture_id = -1;
    int32_t chunk_size = 100;                     // ms
    int32_t max_nomute_chunks = 8000 / chunk_size;// chunks
    int32_t min_mute_chunks = 1000 / chunk_size;  // chunks

    bool use_gpu = true;
    bool flash_attn = false;
    bool debug_mode = false;
    bool use_vad = false;
    bool use_itn = false;
    bool use_prefix = false;
    std::string language = "auto";
    std::string model = "models/ggml-base.en.bin";
    std::string fname_out;
    std::string audio_out;  // 用于输出音频流到文件

    // silero-vad 参数
    float threshold = 0.5f;
    float neg_threshold = 0.35f;
};


void sense_voice_stream_usage(int /*argc*/, char **argv, const sense_voice_stream_params &params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N         [%-7d] [SenseVoice] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "            --chunk_size        [%-7d] vad chunk size(ms)\n", params.chunk_size);
    fprintf(stderr, "  -mmc      --min-mute-chunks   [%-7d] When consecutive chunks are identified as silence\n", params.min_mute_chunks);
    fprintf(stderr, "  -mnc      --max-nomute-chunks [%-7d] when the first non-silent chunk is too far away\n", params.max_nomute_chunks);
    fprintf(stderr, "            --use-vad           [%-7s] when the first non-silent chunk is too far away\n", params.use_vad ? "true" : "false");
    fprintf(stderr, "            --use-prefix        [%-7s] use sense voice prefix\n", params.use_prefix ? "true" : "false");
    fprintf(stderr, "  -c ID,    --capture ID        [%-7d] [Device] capture device ID\n", params.capture_id);
    fprintf(stderr, "  -l LANG,  --language LANG     [%-7s] [SenseVoice] spoken language\n", params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME       [%-7s] [SenseVoice] model path\n", params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME        [%-7s] [IO] text output file name\n", params.fname_out.c_str());
    fprintf(stderr, "  -o FNAME, --output-audio FNAME [%-7s] [IO] audio output file name\n", params.audio_out.c_str());
    fprintf(stderr, "  -ng,      --no-gpu            [%-7s] [SenseVoice] disable GPU inference\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn        [%-7s] [SenseVoice] flash attention during inference\n", params.flash_attn ? "true" : "false");
    fprintf(stderr, "            --use-itn           [%-7s] [SenseVoice] Filter duplicate tokens when outputting\n", params.use_itn ? "true" : "false");
    fprintf(stderr, "            --vad-threshold     [%-7.2f] [VAD] speech probability threshold\n", params.threshold);
    fprintf(stderr, "            --vad-neg-threshold [%-7.2f] [VAD] non-speech probability threshold\n", params.neg_threshold);
    fprintf(stderr, "\n");
}


static bool get_stream_params(int argc, char **argv, sense_voice_stream_params &params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            sense_voice_stream_usage(argc, argv, params);
            exit(0);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--capture") {
            params.capture_id = std::stoi(argv[++i]);
        } else if (arg == "-l" || arg == "--language") {
            params.language = argv[++i];
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-f" || arg == "--file") {
            params.fname_out = argv[++i];
        } else if (arg == "-o" || arg == "--output-audio") {
            params.audio_out = argv[++i];
        } else if (arg == "-ng" || arg == "--no-gpu") {
            params.use_gpu = false;
        } else if (arg == "-fa" || arg == "--flash-attn") {
            params.flash_attn = true;
        } else if (arg == "-debug" || arg == "--debug-mode") {
            params.debug_mode = true;
        } else if (arg == "-mmc" || arg == "--min-mute-chunks") {
            params.min_mute_chunks = std::stoi(argv[++i]);
        } else if (arg == "-mnc" || arg == "--max-nomute-chunks") {
            params.max_nomute_chunks = std::stoi(argv[++i]);
        } else if (arg == "--use-vad") {
            params.use_vad = true;
        } else if (arg == "--use-prefix") {
            params.use_prefix = true;
        } else if (arg == "--chunk-size") {
            params.chunk_size = std::stoi(argv[++i]);
        } else if (arg == "--use-itn") {
            params.use_itn = true;
        } else if (arg == "--vad-threshold") {
            params.threshold = std::stof(argv[++i]);
        } else if (arg == "--vad-neg-threshold") {
            params.neg_threshold = std::stof(argv[++i]);
        }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sense_voice_stream_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void sense_voice_free(struct sense_voice_context *ctx) {
    if (ctx) {
        ggml_free(ctx->model.ctx);

        ggml_backend_buffer_free(ctx->model.buffer);

        sense_voice_free_state(ctx->state);

        delete ctx->model.model->encoder;
        delete ctx->model.model;
        delete ctx;
    }
}


int main(int argc, char **argv) {
    sense_voice_stream_params params;
    if (get_stream_params(argc, argv, params) == false) return 1;

    // VAD 常量定义（与 main.cc 保持一致）
    #define CHUNK_SIZE 512
    #define CONTEXT_SIZE 576
    #define SENSE_VOICE_VAD_CHUNK_PAD_SIZE 64
    #define VAD_LSTM_STATE_MEMORY_SIZE 2048
    #define VAD_LSTM_STATE_DIM 128

    const int n_sample_step = params.chunk_size * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const int keep_nomute_step = params.chunk_size * params.min_mute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;
    const int max_nomute_step = params.chunk_size * params.max_nomute_chunks * 1e-3 * SENSE_VOICE_SAMPLE_RATE;

    audio_async audio(params.chunk_size << 2);
    if (!audio.init(params.capture_id, SENSE_VOICE_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }
    audio.resume();

    struct sense_voice_context_params cparams = sense_voice_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    cparams.use_itn = params.use_itn;

    bool is_running = true;
    struct sense_voice_context *ctx = sense_voice_small_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize sense voice context\n");
        return 3;
    }

    // 设置语言ID（重要：必须设置才能正确识别）
    ctx->language_id = sense_voice_lang_id(params.language.c_str());
    if (ctx->language_id == -1) {
        fprintf(stderr, "warning: unknown language '%s', using auto detection\n", params.language.c_str());
        ctx->language_id = sense_voice_lang_id("auto");
    }

    fprintf(stderr, "Language: %s (ID: %d)\n", params.language.c_str(), ctx->language_id);

    std::vector<float> pcmf32_audio;
    std::vector<double> pcmf32;
    std::vector<double> pcmf32_tmp;// 传递给模型用

    // 预留合理的容量，避免频繁的内存重分配
    pcmf32.reserve(max_nomute_step * 2);
    pcmf32_audio.reserve(params.chunk_size * SENSE_VOICE_SAMPLE_RATE / 1000 * 4); // 预留4个chunk的空间
    pcmf32_tmp.reserve(max_nomute_step); // 为临时处理预留空间

    // 文本输出文件
    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open text output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    // 音频输出文件
    std::ofstream audio_fout;
    WAVHeader wav_header;
    uint32_t total_samples = 0;
    if (params.audio_out.length() > 0) {
        audio_fout.open(params.audio_out, std::ios::binary);
        if (!audio_fout.is_open()) {
            fprintf(stderr, "%s: failed to open audio output file '%s'!\n", __func__, params.audio_out.c_str());
            return 1;
        }
        // 写入 WAV 文件头（先写入占位符，稍后更新）
        audio_fout.write(reinterpret_cast<const char*>(&wav_header), sizeof(WAVHeader));
    }

    // 初始化 VAD 状态（与 main.cc 保持一致）
    if (params.use_vad) {
        // init state
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
    }

    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing samples (chunk = %d ms / max_nomute_chunk = %d / min_mute_chunk = %d), %d threads ...\n",
                __func__,
                params.chunk_size,
                params.max_nomute_chunks,
                params.min_mute_chunks,
                params.n_threads);

        if (!params.use_vad) {
            fprintf(stderr, "%s: not use VAD, will print identified result per %d ms\n", __func__, params.chunk_size * params.max_nomute_chunks);
        } else {
            fprintf(stderr, "%s: using VAD, will print when mute time longer than %d ms or nomute time longer than %d ms\n", __func__, params.chunk_size * params.min_mute_chunks, params.chunk_size * params.max_nomute_chunks);
        }

        fprintf(stderr, "\n");
    }


    sense_voice_full_params wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
    {
        wparams.language = params.language.c_str();
        wparams.n_threads = params.n_threads;
        wparams.debug_mode = params.debug_mode;
    }

    int idenitified_floats = 0, R_new_chunk = 0, L_new_chunk = 0;
    // 只有vad会使用，但是会贯穿全程，只能定义在外面
    std::pair<int, int> nomute = std::pair<int, int>(-1, 0);// 标记最后一段非静音的区间
    std::pair<int, int> mute = std::pair<int, int>(-1, -1); // 标记最后一段静音的区间
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();
        if (!is_running) break;
        // process new audio
        // 获取新的音频，不论是否检测音频数据先把数据捞出来
        std::this_thread::sleep_for(std::chrono::milliseconds(params.chunk_size));
        audio.get(params.chunk_size, pcmf32_audio);

                // 如果需要输出音频到文件，即时写入
        if (audio_fout.is_open() && !pcmf32_audio.empty()) {
            // 转换为 16 位 PCM 格式
            std::vector<int16_t> pcm16_data;
            float_to_pcm16(pcmf32_audio, pcm16_data);

            // 写入 PCM 数据
            audio_fout.write(reinterpret_cast<const char*>(pcm16_data.data()),
                            pcm16_data.size() * sizeof(int16_t));
            audio_fout.flush(); // 确保即时输出

            // 更新样本计数
            total_samples += pcm16_data.size();
        }

        // 转移到pcmf32中，直接识别pcmf32
        pcmf32.insert(pcmf32.end(), pcmf32_audio.begin(), pcmf32_audio.end());
        pcmf32_audio.clear();
        // 新的识别区间：[L_new_chunk, R_new_chunk)，是固定n_sample_step的整数倍
        // fprintf(stderr, "%s || new_audio_size: %d, cache_size: %d, L_new_chunk: %d, R_new_chunk: %d\n", __func__, pcmf32_audio.size(), pcmf32.size(), L_new_chunk, R_new_chunk);
        if (R_new_chunk + n_sample_step <= pcmf32.size() + idenitified_floats) {
            L_new_chunk = R_new_chunk;
            R_new_chunk = int((pcmf32.size() + idenitified_floats) / n_sample_step) * n_sample_step;
        } else
            continue;

        if (!params.use_vad) {
            // 识别当前已经识别到的文本
            printf("\33[2K\r");
            printf("%s", std::string(50, ' ').c_str());
            printf("\33[2K\r");
            printf("[%.2f-%.2f]", idenitified_floats / (SENSE_VOICE_SAMPLE_RATE * 1.0), R_new_chunk / (SENSE_VOICE_SAMPLE_RATE * 1.0));
            if (sense_voice_full_parallel(ctx, wparams, pcmf32, R_new_chunk - idenitified_floats, params.n_processors) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 10;
            }

            sense_voice_print_output(ctx, params.use_prefix, params.use_itn, true);
            // 时间长度太长了直接换行重新开始
            if (R_new_chunk >= max_nomute_step + idenitified_floats) {
                printf("\n");
                // 优化：使用assign而不是重新构造vector
                pcmf32_tmp.assign(pcmf32.begin() + R_new_chunk - idenitified_floats, pcmf32.end());
                pcmf32 = pcmf32_tmp;
                idenitified_floats = R_new_chunk;
            }
        } else {
            // 刷新整行
            printf("\33[2K\r");
            printf("%s", std::string(50, ' ').c_str());
            printf("\33[2K\r");
            // 新进来的所有chunk有可能导致序列分拆，需要注意
            for (int i = L_new_chunk; i < R_new_chunk; i += n_sample_step) {
                // int R_this_chunk = i + n_sample_step;
                // 使用 silero-vad 替换 vad_energy_zcr
                bool isnomute = false;

                                // 准备 CONTEXT_SIZE 的 chunk 数据，与 main.cc 保持一致
                std::vector<float> chunk(CONTEXT_SIZE + SENSE_VOICE_VAD_CHUNK_PAD_SIZE, 0);
                int start_idx = i - idenitified_floats;

                // 修复：使用更小的窗口进行VAD检测，避免超出边界
                int vad_chunk_start = std::max(0, start_idx);
                int vad_chunk_end = std::min(static_cast<int>(pcmf32.size()), vad_chunk_start + CONTEXT_SIZE);

                for (int j = 0; j < CONTEXT_SIZE; j++) {
                    if (vad_chunk_start + j < vad_chunk_end) {
                        // 注意：pcmf32是double类型，需要正确转换
                        chunk[j] = static_cast<float>(pcmf32[vad_chunk_start + j]);
                    } else {
                        chunk[j] = 0;
                    }
                }
                // 实现反射填充
                for (int j = CONTEXT_SIZE; j < chunk.size(); j++) {
                    int reflect_idx = 2 * CONTEXT_SIZE - j - 2;
                    if (reflect_idx >= 0 && reflect_idx < CONTEXT_SIZE) {
                        chunk[j] = chunk[reflect_idx];
                    } else {
                        chunk[j] = 0;
                    }
                }

                float speech_prob = 0;
                if (silero_vad_encode_internal(*ctx, *ctx->state, chunk, params.n_threads, speech_prob)) {
                    isnomute = (speech_prob >= params.threshold);
                    // 调试信息：显示VAD结果
                    // if (speech_prob > 0.1) { // 只显示有意义的概率
                    //     fprintf(stderr, "VAD: prob=%.3f, threshold=%.3f, isnomute=%d\n",
                    //            speech_prob, params.threshold, isnomute);
                    // }
                } else {
                    // fprintf(stderr, "VAD failed, using energy detection\n");
                    // 如果 VAD 处理失败，回退到简单的能量检测
                    double energy = 0;
                    for (int j = 0; j < n_sample_step && start_idx + j < pcmf32.size(); j++) {
                        if (start_idx + j >= 0) {
                            energy += pcmf32[start_idx + j] * pcmf32[start_idx + j];
                        }
                    }
                    energy /= n_sample_step;
                    isnomute = (energy > 1e-3); // 调整能量阈值
                }
                // fprintf(stderr, "Mute || isnomute = %d, ML = %d, MR = %d, NML = %d, NMR = %d, R_new_chunk = %d, i = %d, size = %d, idenitified = %d\n", isnomute, mute.first, mute.second, nomute.first, nomute.second, R_new_chunk, i, pcmf32.size(), idenitified_floats);
                if (nomute.first == -1) {
                    if (isnomute) nomute.first = i;
                    continue;
                }
                if (mute.first == -1) {
                    // type1 [NML, ...)
                    if (!isnomute) mute.first = i;
                } else if (mute.second == -1) {
                    // type2 [NML, ML), [ML, ...)
                    if (isnomute) mute.second = i;
                    else if (i - mute.first >= keep_nomute_step) {
                        // 需要识别[NML, ML)，并退化到nomute, mute全部为-1的情况
                        pcmf32_tmp.resize(mute.first - nomute.first);
                        std::copy(pcmf32.begin() + nomute.first - idenitified_floats, pcmf32.begin() + mute.first - idenitified_floats, pcmf32_tmp.begin());
                        printf("[%.2f-%.2f]", nomute.first / (SENSE_VOICE_SAMPLE_RATE * 1.0), mute.first / (SENSE_VOICE_SAMPLE_RATE * 1.0));
                        if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), params.n_processors) != 0) {
                            fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                            return 10;
                        }
                        sense_voice_print_output(ctx, params.use_prefix, params.use_itn);// 这里整行输出即可
                        nomute.second = i;
                        nomute.first = mute.first = mute.second = -1;
                    }
                } else {
                    // type3 [NML, ML), [ML, MR), [MR, ...)
                    if (!isnomute) mute.first = i, mute.second = -1;
                }
                // 可能需要裂解
                if (nomute.first >= 0 && i - nomute.first >= max_nomute_step) {
                    int R_nomute = mute.first == -1 ? nomute.first + max_nomute_step : mute.first;
                    pcmf32_tmp.resize(R_nomute - nomute.first);
                    std::copy(pcmf32.begin() + (nomute.first - idenitified_floats), pcmf32.begin() + (R_nomute - idenitified_floats), pcmf32_tmp.begin());
                    printf("[%.2f-%.2f]", nomute.first / (SENSE_VOICE_SAMPLE_RATE * 1.0), R_nomute / (SENSE_VOICE_SAMPLE_RATE * 1.0));
                    if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), params.n_processors) != 0) {
                        fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                        return 10;
                    }
                    sense_voice_print_output(ctx, params.use_prefix, params.use_itn);// 这里整行输出即可
                    if (mute.first == -1) {
                        nomute.second = nomute.first + max_nomute_step;
                        nomute.first += max_nomute_step;
                    } else if (mute.second == -1) {
                        nomute.second = mute.first;
                        nomute.first = mute.first = mute.second = -1;
                    } else {
                        nomute.second = mute.first;
                        nomute.first = mute.second;
                        mute.first = mute.second = -1;
                    }
                }
            }
            // 输出最后一段
            if (nomute.first >= 0) {
                pcmf32_tmp.resize(R_new_chunk - nomute.first);
                std::copy(pcmf32.begin() + (nomute.first - idenitified_floats), pcmf32.begin() + (R_new_chunk - idenitified_floats), pcmf32_tmp.begin());
                printf("[%.2f-%.2f]", nomute.first / (SENSE_VOICE_SAMPLE_RATE * 1.0), R_new_chunk / (SENSE_VOICE_SAMPLE_RATE * 1.0));
                if (sense_voice_full_parallel(ctx, wparams, pcmf32_tmp, pcmf32_tmp.size(), params.n_processors) != 0) {
                    fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                    return 10;
                }
                sense_voice_print_output(ctx, params.use_prefix, params.use_itn, true);// 这里整行输出即可
            }
            // 调整idenitified_floats并且减少pcmf32的长度
            if (nomute.second > 0) {
                // 优化：使用assign而不是重新构造vector
                pcmf32_tmp.assign(pcmf32.begin() + (nomute.second - idenitified_floats), pcmf32.end());
                pcmf32 = pcmf32_tmp;
                idenitified_floats = nomute.second;
                nomute.second = 0;
            } else if (nomute.first == -1) {
                // 优化：使用assign而不是重新构造vector
                pcmf32_tmp.assign(pcmf32.begin() + (R_new_chunk - idenitified_floats), pcmf32.end());
                pcmf32 = pcmf32_tmp;
                idenitified_floats = R_new_chunk;
            }

            // 检查缓冲区大小并发出警告
            if (pcmf32.size() > 2 * max_nomute_step) {
                fprintf(stderr, "Warning: Audio buffer size (%.2f MB, %.2f sec) exceeds recommended limit. Consider optimizing processing speed.\n",
                       pcmf32.size() * sizeof(double) / 1e6,
                       pcmf32.size() / (double)SENSE_VOICE_SAMPLE_RATE);
            }
        }
        fflush(stdout);
    }
    audio.pause();

    // 关闭输出文件
    if (fout.is_open()) {
        fout.close();
    }
    if (audio_fout.is_open()) {
        // 更新 WAV 文件头中的文件大小信息
        wav_header.data_size = total_samples * sizeof(int16_t);
        wav_header.file_size = sizeof(WAVHeader) - 8 + wav_header.data_size;

        // 重新定位到文件开头并写入更新后的文件头
        audio_fout.seekp(0, std::ios::beg);
        audio_fout.write(reinterpret_cast<const char*>(&wav_header), sizeof(WAVHeader));
        audio_fout.close();

        fprintf(stderr, "Audio saved to '%s' (%u samples, %.2f seconds)\n",
                params.audio_out.c_str(), total_samples,
                (float)total_samples / SENSE_VOICE_SAMPLE_RATE);
    }

    sense_voice_free(ctx);
    return 0;
}
