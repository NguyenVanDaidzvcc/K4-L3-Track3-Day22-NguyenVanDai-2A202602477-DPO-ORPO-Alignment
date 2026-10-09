
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB0 — DPO loss tự cài từ đầu (CPU, ~10 phút)
#
# **Không cần GPU.** Trước khi gọi `DPOTrainer`, bạn tự viết loss
# và kiểm tra nó trên số liệu đồ chơi.
# Phần này lấy từ lab K3 (tự cài DPO) và là nền để đọc
# đường cong reward ở NB3.
#
# Bạn sẽ thấy:
# 1. Tại bước 0 (policy = reference),
#    loss luôn bằng log 2 ≈ 0.693.
# 2. Gradient của DPO bị nhân với sigmoid(-margin):
#    cặp đã phân biệt tốt gần như không còn được học.
# 3. Likelihood displacement: loss vẫn giảm khi log-prob
#    của chosen giảm, miễn rejected giảm nhanh hơn.
# 4. IPO, RPO, SimPO, ORPO khác DPO ở đâu,
#    trên cùng một bộ số.


# %%
import sys
from pathlib import Path

ROOT = next(
    p for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "lab22" / "config.py").exists()
)

sys.path.insert(0, str(ROOT))

import math
import torch

from lab22 import dpo_math as M

torch.manual_seed(0)


# %% [markdown]
# ## 1. Log-prob của một câu trả lời
#
# log pi(y|x) = sum_t log pi(y_t | x, y_<t)
#
# Chỉ cộng trên token của câu trả lời (mask = 1),
# không cộng trên câu hỏi.


# %%
vocab, length = 8, 5

logits = torch.randn(1, length, vocab)

labels = torch.randint(
    0,
    vocab,
    (1, length)
)

# 2 token prompt, 3 token trả lời
mask = torch.tensor([[0, 0, 1, 1, 1]])

total, mean = M.sequence_logps(
    logits,
    labels,
    mask
)

print(
    f"sum log p = {total.item():.3f}   "
    f"mean log p = {mean.item():.3f}"
)


# %% [markdown]
# ## 2. Bài tập: tự viết DPO loss
#
# Công thức (Rafailov et al. 2023):
#
# L = -log sigmoid(
#     beta * (
#         (log policy_chosen - log reference_chosen)
#       - (log policy_rejected - log reference_rejected)
#     )
# )
#
# Hàm bên dưới sử dụng torch.nn.functional.logsigmoid.
# So sánh với bản tham chiếu trong lab22/dpo_math.py.


# %%
def my_dpo_loss(pc, pr, rc, rr, beta=0.1):
    """
    Tính DPO Loss từ đầu.

    Parameters
    ----------
    pc : torch.Tensor
        Policy log-prob của chosen.

    pr : torch.Tensor
        Policy log-prob của rejected.

    rc : torch.Tensor
        Reference log-prob của chosen.

    rr : torch.Tensor
        Reference log-prob của rejected.

    beta : float
        Hệ số điều chỉnh DPO.

    Returns
    -------
    torch.Tensor
        DPO loss trung bình.
    """

    # Bước 1: Tính log-ratio của policy
    policy_logratios = pc - pr

    # Bước 2: Tính log-ratio của reference
    reference_logratios = rc - rr

    # Bước 3: Tính preference margin
    preference_margin = (
        policy_logratios - reference_logratios
    )

    # Bước 4: Nhân với beta
    logits = beta * preference_margin

    # Bước 5: DPO objective
    loss = -torch.nn.functional.logsigmoid(logits)

    # Bước 6: Trả về loss trung bình
    return loss.mean()


# %% [markdown]
# ### Kiểm tra hàm DPO Loss
#
# So sánh hàm tự viết với hàm tham chiếu.
# Nếu chênh lệch quá 1e-6, chương trình sẽ báo lỗi.


# %%
pc = torch.tensor([-12.0, -30.0])
pr = torch.tensor([-15.0, -28.0])

rc = torch.tensor([-13.0, -29.0])
rr = torch.tensor([-14.0, -29.0])

# Loss tham chiếu
ref_loss, _, _ = M.dpo_loss(
    pc,
    pr,
    rc,
    rr,
    beta=0.1
)

# Loss tự cài đặt
mine = my_dpo_loss(
    pc,
    pr,
    rc,
    rr,
    beta=0.1
)

if mine is None:
    print(
        "Chưa cài my_dpo_loss. "
        f"Đáp số tham chiếu: {ref_loss.item():.4f}"
    )

else:
    assert torch.allclose(
        torch.as_tensor(mine),
        ref_loss,
        atol=1e-6
    ), (mine, ref_loss)

    print(
        f"✓ Khớp tham chiếu: {ref_loss.item():.4f}"
    )


# %% [markdown]
# ## 3. Bước 0: policy = reference
#
# Khi mô hình đang học bằng mô hình tham chiếu,
# preference margin = 0.
#
# DPO loss = -log(sigmoid(0)) = log(2) ≈ 0.693.
#
# NB3 khởi tạo policy bằng mô hình SFT.
# Nếu loss ban đầu không gần 0.693, cần kiểm tra
# reference có đúng là mô hình SFT hay không.


# %%
same = torch.tensor([-20.0, -35.0])

loss0, cr0, rr0 = M.dpo_loss(
    same,
    same - 3,
    same,
    same - 3
)

print(
    f"loss at init = {loss0.item():.4f}   "
    f"log 2 = {math.log(2):.4f}   "
    f"rewards = {cr0.tolist()}, {rr0.tolist()}"
)

# Kiểm tra giá trị khởi tạo
assert abs(
    loss0.item() - math.log(2)
) < 1e-6

print("✓ Initial DPO loss = log(2)")


# %% [markdown]
# ## 4. Trọng số gradient = sigmoid(-margin)
#
# DPO tập trung cập nhật các preference pair
# mà mô hình chưa phân biệt được tốt.
#
# Khi margin lớn và dương, gradient nhỏ.
# Khi margin âm, gradient lớn hơn.


# %%
for margin in (-2.0, 0.0, 2.0, 5.0):

    m = torch.tensor(
        margin,
        requires_grad=True
    )

    loss = -torch.nn.functional.logsigmoid(m)

    loss.backward()

    print(
        f"margin {margin:+.1f}: "
        f"loss {loss.item():.3f}   "
        f"|dL/dmargin| {abs(m.grad.item()):.3f}"
    )


# %% [markdown]
# ## 5. Likelihood displacement bằng số
#
# Hai kịch bản đều làm margin tăng 2 nat.
#
# Kịch bản A:
# - Chosen log-prob tăng.
# - Rejected log-prob giảm.
#
# Kịch bản B:
# - Chosen log-prob giảm.
# - Rejected log-prob giảm mạnh hơn.
#
# Trong hai trường hợp, preference margin bằng nhau,
# nên DPO loss bằng nhau.
#
# Đây là hiện tượng likelihood displacement.


# %%
ref_c = torch.tensor([-20.0])
ref_r = torch.tensor([-22.0])

scenarios = {
    "A: chosen ↑, rejected ↓": (
        ref_c + 1,
        ref_r - 1
    ),

    "B: chosen ↓, rejected ↓↓": (
        ref_c - 3,
        ref_r - 5
    ),
}

for name, (pc_, pr_) in scenarios.items():

    loss, cr, rj = M.dpo_loss(
        pc_,
        pr_,
        ref_c,
        ref_r,
        beta=1.0
    )

    print(
        f"{name:28s} "
        f"loss {loss.item():.3f}  "
        f"reward chosen {cr.item():+.1f}  "
        f"rejected {rj.item():+.1f}"
    )


# %% [markdown]
# ### So sánh với RPO
#
# RPO bổ sung NLL loss của chosen vào DPO loss.
#
# Mục tiêu là hạn chế việc mô hình giảm xác suất
# của chosen quá nhiều trong quá trình căn chỉnh.


# %%
for name, (pc_, pr_) in scenarios.items():

    # NLL trung bình trên 10 token
    nll = -pc_ / 10

    rpo = M.rpo_loss(
        pc_,
        pr_,
        ref_c,
        ref_r,
        nll,
        beta=1.0
    )

    print(
        f"{name:28s} "
        f"RPO loss {rpo.item():.3f}"
    )


# %% [markdown]
# ## 6. Bốn biến thể trên cùng một cặp
#
# | Loss | Reference? | Chuẩn hoá độ dài? |
# |------|------------|-------------------|
# | DPO  | Có         | Không             |
# | IPO  | Có         | Có                |
# | RPO  | Có         | Không             |
# | SimPO| Không      | Có                |
# | ORPO | Không      | Có                |
#
# DPO:
#   Tối ưu trực tiếp preference giữa chosen và rejected.
#
# IPO:
#   Hạn chế quá khớp khi dữ liệu preference gần tất định.
#
# RPO:
#   Kết hợp DPO và NLL của chosen.
#
# SimPO:
#   Dùng log-prob trung bình trên token và margin gamma.
#
# ORPO:
#   Kết hợp NLL và odds-ratio.
#
# NB3b sẽ huấn luyện thật các biến thể.


# %%
# Chosen ngắn, rejected dài
n_tokens_c = 40
n_tokens_r = 120

# Log-prob của policy
pc_ = torch.tensor([-48.0])
pr_ = torch.tensor([-130.0])

# Log-prob của reference
rc_ = torch.tensor([-50.0])
rr_ = torch.tensor([-128.0])

# Chuẩn hoá log-prob theo độ dài
avg_c = pc_ / n_tokens_c
avg_r = pr_ / n_tokens_r

# DPO
dpo_loss = M.dpo_loss(
    pc_,
    pr_,
    rc_,
    rr_
)[0]

# IPO
ipo_loss = M.ipo_loss(
    pc_,
    pr_,
    rc_,
    rr_,
    n_tokens_c,
    n_tokens_r
)

# SimPO
simpo_loss = M.simpo_loss(
    avg_c,
    avg_r
)

# ORPO
orpo_loss = M.orpo_loss(
    avg_c,
    avg_r,
    -avg_c
)

print("\n=== Preference Loss Comparison ===")

print(f"DPO   {dpo_loss.item():.4f}")
print(f"IPO   {ipo_loss.item():.4f}")
print(f"SimPO {simpo_loss.item():.4f}")
print(f"ORPO  {orpo_loss.item():.4f}")


# %% [markdown]
# ## 7. Câu hỏi REFLECTION §3
#
# Tổng log-prob của một chuỗi là tổng các
# log-xác suất token. Vì mỗi log-xác suất
# không dương, câu dài thường có tổng log-prob
# âm hơn câu ngắn.
#
# DPO gốc dùng tổng log-prob của chosen và rejected.
# Nếu hai câu có độ dài khác nhau, chênh lệch
# log-prob có thể chịu ảnh hưởng mạnh bởi độ dài.
#
# Điều này có thể khiến mô hình học cả xu hướng
# liên quan đến độ dài thay vì chỉ học chất lượng.
#
# SimPO sử dụng log-prob trung bình trên mỗi token,
# giúp giảm ảnh hưởng trực tiếp của độ dài chuỗi.
#
# ORPO cũng dùng log-prob trung bình theo token
# trong phần preference objective, đồng thời
# bổ sung NLL của chosen.
#
# Việc chuẩn hoá độ dài giúp giảm thiên vị,
# nhưng không đảm bảo loại bỏ hoàn toàn
# ảnh hưởng của độ dài câu trả lời.


# %%
print("\n" + "=" * 55)
print("NB0 - DPO LOSS FROM SCRATCH")
print("=" * 55)
print("DPO implementation: COMPLETED")
print("Reference loss check: PASSED")
print("Initialization check: PASSED")
print("Gradient analysis: COMPLETED")
print("Likelihood displacement: COMPLETED")
print("RPO comparison: COMPLETED")
print("Preference loss comparison: COMPLETED")
print("=" * 55)
print("NB0 FINISHED SUCCESSFULLY!")
