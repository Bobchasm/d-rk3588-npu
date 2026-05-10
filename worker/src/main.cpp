#include "qwen2.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/time.h>

static int64_t now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

// ============================================================
// 用法：
//   ./qwen2_demo <model_dir> <token_id1> [token_id2 ...]
//
// 例：先用 Python 获取 token id，再传入
//   python3 -c "from transformers import AutoTokenizer; \
//     t=AutoTokenizer.from_pretrained('model/Qwen1.5B'); \
//     ids=t.encode('你好'); print(' '.join(map(str,ids)))"
//   # 假设输出：151644 872 1773 151645
//   ./qwen2_demo ../model/Qwen1.5B 151644 872 1773 151645
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <token_id> [token_id ...]\n", argv[0]);
        return 1;
    }

    std::string model_dir = argv[1];
    std::vector<int> input_ids;
    for (int i = 2; i < argc; ++i)
        input_ids.push_back(atoi(argv[i]));

    printf("模型目录: %s\n", model_dir.c_str());
    printf("输入 token 数量: %d\n", (int)input_ids.size());
    for (int id : input_ids) printf("  %d\n", id);

    // 加载模型
    Qwen2Model model;
    printf("\n开始加载模型...\n");
    int64_t t0 = now_us();
    if (!model.load(model_dir)) {
        fprintf(stderr, "模型加载失败\n");
        return 1;
    }
    printf("模型加载完成，耗时 %.1f s\n", (now_us() - t0) / 1e6f);

    // 推理 + 贪心生成最多 50 个 token
    const int max_new_tokens = 10;
    std::vector<int> all_tokens = input_ids;

    printf("\n开始推理...\n");
    int64_t t_infer_start = now_us();
    int generated = 0;
    for (int step = 0; step < max_new_tokens; ++step) {
        int64_t ts = now_us();

        // 每步都传完整 token 序列（无 KV Cache，正确但慢）
        // TODO: 接入 KV Cache 后改成增量 decode
        std::vector<int> cur_input = all_tokens;

        auto logits = model.forward(cur_input);
        int next_id = greedy_sample(logits);

        int64_t elapsed = now_us() - ts;

        // 打印 top-5，便于验证是否输出合理
        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < (int)logits.size(); ++i)
            top.push_back({logits[i], i});
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
            [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; });
        printf("step %2d (%.0f ms): top5 = ", step, elapsed / 1e3f);
        for (int i = 0; i < 5; ++i)
            printf("[%d:%.2f] ", top[i].second, top[i].first);
        printf("-> next=%d\n", next_id);

        // 检查 EOS（Qwen 的 EOS id = 151645）
        if (next_id == 151645 || next_id == 151643) {
            printf("[EOS]\n");
            break;
        }

        all_tokens.push_back(next_id);
        ++generated;
    }

    float total_s = (now_us() - t_infer_start) / 1e6f;
    printf("\n生成完成：%d tokens，总耗时 %.1f s，平均 %.2f tok/s\n",
           generated, total_s, generated / total_s);

    printf("\n输出 token ids:");
    for (int i = (int)input_ids.size(); i < (int)all_tokens.size(); ++i)
        printf(" %d", all_tokens[i]);
    printf("\n");

    return 0;
}
