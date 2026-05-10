# inspect_model.py
from transformers import AutoConfig
import torch
import json, os

model_path = "../model/Qwen1.5B"  # 你下载的路径

# 1. 看配置
config = AutoConfig.from_pretrained(model_path)
print("=" * 50)
print("模型配置:")
print(f"  架构:         {config.architectures}")
print(f"  层数:         {config.num_hidden_layers}")
print(f"  隐藏层维度:    {config.hidden_size}")
print(f"  注意力头数:    {config.num_attention_heads}")
print(f"  KV头数:       {config.num_key_value_heads}  (GQA)")
print(f"  每头维度:      {config.hidden_size // config.num_attention_heads}")
print(f"  中间层维度:    {config.intermediate_size}")
print(f"  激活函数:      {config.hidden_act}")
print(f"  词表大小:      {config.vocab_size}")
print(f"  最大序列长度:  {config.max_position_embeddings}")
print("=" * 50)

# 2. 推导每层的 MatMul shape（不加载权重）
H = config.hidden_size
I = config.intermediate_size
NH = config.num_attention_heads
NKV = config.num_key_value_heads
HD = H // NH  # head_dim
KV_DIM = NKV * HD  # K/V 投影后的总维度

print("\n每个 Transformer Block 中的 Linear（MatMul）:")
print(f"  [Attention]")
print(f"    q_proj:   ({H}) -> ({H})       = M x {H} @ {H} x {H}")
print(f"    k_proj:   ({H}) -> ({KV_DIM})  = M x {H} @ {H} x {KV_DIM}")
print(f"    v_proj:   ({H}) -> ({KV_DIM})  = M x {H} @ {H} x {KV_DIM}")
print(f"    o_proj:   ({H}) -> ({H})       = M x {H} @ {H} x {H}")
print(f"    QK^T:     M x {HD} @ {HD} x seqlen  (per head, 12 heads)")
print(f"    score@V:  M x seqlen @ seqlen x {HD}  (per head, 12 heads)")
print(f"  [FFN (SwiGLU)]")
print(f"    gate_proj: ({H}) -> ({I}) = M x {H} @ {H} x {I}")
print(f"    up_proj:   ({H}) -> ({I}) = M x {H} @ {H} x {I}")
print(f"    down_proj: ({I}) -> ({H}) = M x {I} @ {I} x {H}")
print()
print("K对齐检查（必须是32的倍数）:")
for name, k in [("hidden_size", H), ("KV_DIM", KV_DIM), ("intermediate_size", I), ("head_dim", HD)]:
    aligned = "✅" if k % 32 == 0 else "❌"
    print(f"  {name}={k}: {k}/32={k//32}余{k%32}  {aligned}")

print()
print("每层MatMul数量: 7个 (q/k/v/o/gate/up/down)")
print(f"总MatMul调用次数: 7 × {config.num_hidden_layers} = {7*config.num_hidden_layers} 次/token")

# 3. 用 safetensors metadata 看权重分布（不加载到内存）
try:
    from safetensors import safe_open
    sf_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(sf_path):
        print("\n权重文件元信息（不加载到内存）:")
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        # 只打印第0层的key
        layer0 = [k for k in keys if "layers.0." in k]
        print(f"  第0层权重 keys ({len(layer0)}个):")
        for k in layer0:
            print(f"    {k}")
except ImportError:
    print("(safetensors 未安装，跳过权重检查)")
except Exception as e:
    print(f"(权重检查失败: {e})")