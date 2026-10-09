# Bài phản tư — Lab 22: DPO/ORPO Alignment (phần bắt buộc)

**Tên:** Nguyễn Văn Đại  
**Mã học viên:** 2A202602477  
**Khoá:** K4 — Track 3 — Ngày 22  
**Tier thực nghiệm:** T4  
**Ngày hoàn thiện:** 08/10/2026

> **Trạng thái:** Bản điền dựa trên dữ liệu đã xác nhận từ Colab và cấu hình Lab. Những trường ghi **[CẦN ĐIỀN TỪ FILE KẾT QUẢ]** chưa thể kết luận vì chưa có `adapters/dpo/dpo_metrics.json` và `data/eval/judge_summary.json` của lần chạy hiện tại. Không tự suy đoán kết quả DPO hoặc giám khảo. Chỉ nộp sau khi đã đối chiếu các mục này và hoàn thành NB4.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab Tesla T4, VRAM danh định 16 GB (Unsloth hiển thị khoảng 14,56 GB khả dụng) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch, chỉ tính loss trên token phản hồi |
| SFT / LoRA | LoRA r = 16; alpha = 32; lưu adapter và mô hình SFT merged 16-bit |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt), cấu hình 800 train / 100 held-out, chia theo câu hỏi không trùng |
| Tỉ lệ chosen dài hơn rejected (NB2) | **66%** (đọc từ biểu đồ `02b-pref-length.png`) |
| DPO: β / lr / epoch | **0,1 / 5e-6 / 1** (cấu hình T4 trong `lab22/config.py`; cần xác nhận không bị override khi train) |
| Giám khảo | **[CẦN ĐIỀN TỪ FILE KẾT QUẢ]**; cấu hình mặc định là panel hai reward model Skywork V2 |
| Chi phí thực tế | **[CẦN XÁC NHẬN]** (không suy ra từ việc dùng Colab) |
| Seed | 42 |

**Kết quả kiểm tra NB0:** Hàm DPO tự cài đặt cho loss **0,6981**, trùng hàm tham chiếu; khi policy trùng reference thì loss bằng log(2) ≈ 0,6931. Hai kịch bản likelihood displacement ở NB0 có thể cho cùng loss dù reward của chosen khác hướng.  
**Kết quả SFT (NB1):** Final SFT training loss = **1,3602**. Biểu đồ `02-sft-loss.png` cho thấy loss giảm rõ từ các bước đầu, dù có dao động cục bộ; đó là loss trên tập huấn luyện, không đủ để kết luận mô hình sinh câu trả lời tốt hơn.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | **[CẦN ĐIỀN TỪ LOG NB3]** |
| VRAM cao nhất | **[CẦN ĐIỀN TỪ LOG GPU]** |
| DPO `final_train_loss` | **[CẦN ĐIỀN TỪ `dpo_metrics.json`]** |
| DPO `first_logged_loss` | **[CẦN ĐIỀN TỪ `dpo_metrics.json`]** |
| Reward chosen cuối trên train | **[CẦN ĐIỀN TỪ `end_chosen_reward`]** |
| Reward rejected cuối trên train | **[CẦN ĐIỀN TỪ `end_rejected_reward`]** |
| Reward gap cuối trên train | **[CẦN ĐIỀN TỪ `end_reward_gap`]** |
| Reward accuracy held-out | **[CẦN ĐIỀN TỪ `eval_reward_accuracy`]** |
| Reward margin held-out | **[CẦN ĐIỀN TỪ `eval_reward_gap`]** |
| Chẩn đoán tự động | **[CẦN ĐIỀN TỪ `diagnosis`]** |
| Độ dài trung bình SFT → DPO | **[CẦN ĐIỀN TỪ `judge_summary.json`]** |

**Diễn giải sơ bộ:** Trong DPO, reward ngầm được tính theo β nhân với chênh lệch log-prob giữa policy và reference. Vì vậy margin dương chỉ nói policy xếp chosen cao hơn rejected *tương đối* so với reference; không đủ chứng minh chất lượng câu trả lời thực tế tốt hơn. Cần đọc đồng thời chosen, rejected và đánh giá trên held-out.

---

## 3. Đọc đường reward (ít nhất 100 từ)

Ảnh thực nghiệm: `submission/screenshots/03-dpo-reward-curves.png`.

Tôi dùng mô hình SFT đã gộp làm reference và khởi tạo LoRA DPO mới làm policy để tránh nhầm lẫn với mô hình gốc. Tại thời điểm khởi đầu, khi policy chưa cập nhật, reward ngầm của chosen và rejected về lý thuyết gần bằng 0. Trong quá trình huấn luyện, tôi cần đọc **riêng** đường `rewards/chosen` và `rewards/rejected` của cả train lẫn held-out, thay vì chỉ nhìn margin. Nếu reward chosen tăng còn reward rejected giảm thì margin tăng theo hướng mong đợi. Ngược lại, chosen vẫn có thể giảm trong khi rejected giảm nhanh hơn, làm margin tăng dù policy gán xác suất thấp hơn cho cả hai câu trả lời; đây là hiện tượng likelihood displacement mà bài NB0 đã minh họa bằng hai trường hợp cùng DPO loss. Dữ liệu NB2 cũng cho thấy chosen dài hơn rejected trong 66% cặp, nên cần cảnh giác mô hình học xu hướng trả lời dài. Sự chênh lệch giữa đường train và held-out là dấu hiệu quan trọng để nhận xét về khả năng tổng quát hóa, tránh nhầm cải thiện trên tập huấn luyện với cải thiện thực chất.

**Kết luận về lần chạy cụ thể — [CẦN ĐIỀN SAU KHI ĐỐI CHIẾU ẢNH VÀ JSON]:** Reward chosen train thay đổi **[...]**, rejected train **[...]**; chosen held-out **[...]**, rejected held-out **[...]**. Margin held-out cuối **[...]**; `diagnosis` là **[...]**. Tôi đánh giá các đường train và held-out **[...]**, vì **[...]**. Chưa thể kết luận INTENDED hay LIKELIHOOD DISPLACEMENT chỉ dựa vào biểu đồ SFT hoặc tỉ lệ độ dài của NB2.

---

## 4. So sánh SFT với SFT+DPO

Ảnh thực nghiệm: `submission/screenshots/04-side-by-side-table.png`. Bảng thống kê phải lấy đúng từ `data/eval/judge_summary.json` sau khi chấm các câu hỏi.

| Nhóm | n | DPO thắng | SFT thắng | Hòa | DPO win rate (CI 95%) | Win rate khi độ dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---|---|
| Held-out | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** |
| Hữu ích (4 câu) | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** |
| An toàn (4 câu) | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** | **[JSON]** |

- Giám khảo thực tế: **[CẦN ĐIỀN: `judge`]**.
- Sanity accuracy: **[CẦN ĐIỀN: `sanity_accuracy`]**.
- Các giám khảo riêng (`per_judge`): **[CẦN ĐIỀN]**.
- Tương quan điểm với độ dài (`score_length_spearman`) / position consistency: **[CẦN ĐIỀN]**.
- Độ dài đầu ra trung bình: SFT **[...]** → DPO **[...]** ký tự.

**Cách nhận xét:** Nếu khoảng tin cậy 95% của win rate chứa 0,5 thì chưa đủ bằng chứng kết luận DPO tốt hơn SFT. Nếu sanity accuracy dưới 0,8, độ tin cậy của giám khảo tiếng Việt bị hạn chế. Nếu câu dài hơn thường thắng hoặc tỷ lệ thắng giảm rõ trên nhóm độ dài gần bằng, có khả năng mô hình hoặc giám khảo bị thiên vị độ dài. Khi panel có hai giám khảo, cần so `per_judge` để kiểm tra mức bất đồng và rủi ro preference leakage; không coi một giám khảo là chân lý tuyệt đối.

**Hai ví dụ cụ thể cần hoàn thiện từ `data/eval/side_by_side.jsonl` và kết quả chấm:**

1. **Hữu ích — h1 (quicksort):** Nội dung SFT thực tế **[TRÍCH NGẮN]**; nội dung DPO **[TRÍCH NGẮN]**; nhận xét về độ chính xác, độ rõ ràng, mức tuân thủ yêu cầu 5–7 câu và kết quả chấm **[ĐIỀN]**.
2. **An toàn — s4 (áp lực thi cử):** Nội dung SFT thực tế **[TRÍCH NGẮN]**; nội dung DPO **[TRÍCH NGẮN]**; nhận xét về sự từ chối hỗ trợ gây hại, cách định hướng sang hỗ trợ an toàn và kết quả chấm **[ĐIỀN]**.

**Kết luận NB4 — [CẦN ĐIỀN]:** Sau khi so win rate, CI 95%, sanity, length-matched và hai ví dụ, tôi rút ra **[...]**. Biểu đồ 8 câu đơn lẻ không thay thế được phép đánh giá ≥ 50 câu held-out.

---

## 6. Một quyết định quan trọng nhất (ít nhất 150 từ)

**Quyết định:** Sử dụng DPO sigmoid loss với β = 0,1 và learning rate 5e-6 trên LoRA r = 16, thay vì tăng β hoặc tinh chỉnh toàn bộ mô hình. Đây là lựa chọn cấu hình được dùng trong phần hướng dẫn T4 của bài lab; khi báo cáo kết quả thực tế tôi cần xác minh rằng không có biến môi trường nào ghi đè các giá trị đó.

Tôi cân nhắc một số phương án khác. Nếu giảm β xuống 0,05, mô hình được phép dịch chuyển mạnh hơn so với reference, nhưng có thể dễ tối ưu quá mức trên các cặp preference. Nếu tăng β lên 0,5, mức phạt khi thay đổi xa reference mạnh hơn, có thể giúp hạn chế dịch chuyển nhưng cũng có thể khiến phần cải thiện khó quan sát trong số bước huấn luyện có hạn. Tương tự, learning rate thấp hơn như 5e-7 sẽ cập nhật LoRA nhẹ hơn, nhưng có nguy cơ đường reward gần như đứng yên trong một thí nghiệm ngắn. Tôi chọn β = 0,1 và 5e-6 vì đó là điểm xuất phát có lý do rõ ràng, phù hợp GPU T4 và đủ đơn giản để giải thích quan hệ giữa policy, reference và margin.

Hai kết quả đã quan sát trước DPO là final SFT loss bằng 1,3602 và tỷ lệ chosen dài hơn rejected ở NB2 là 66%. Kết quả thứ nhất cho thấy bài toán SFT đã có tín hiệu học trên dữ liệu huấn luyện; kết quả thứ hai cảnh báo thiên vị độ dài. Chúng **không** chứng minh rằng cấu hình DPO được chọn sẽ thắng SFT. Bằng chứng xác nhận hay phản bác quyết định phải đến từ `dpo_metrics.json` và `judge_summary.json`, nhất là độ chính xác reward held-out, win rate có khoảng tin cậy 95% và length-matched win rate.

Nếu được thực hiện lại với nhiều GPU hơn, tôi sẽ giữ nguyên cách chia train/held-out và các câu hỏi đánh giá, sau đó thử β = 0,05 và 0,5 dưới cùng điều kiện, đo mức thay đổi chosen/rejected và chất lượng đầu ra. Tôi cũng sẽ lưu checkpoint cùng mọi JSON, PNG ngay sau mỗi giai đoạn để tránh mất bằng chứng khi Colab kết thúc phiên.

**Đánh giá cuối về lựa chọn này — [CẦN ĐIỀN]:** DPO train loss **[...]**, held-out accuracy **[...]**, win rate **[...]**. Các số liệu này **[ủng hộ / không đủ chứng cứ / phản bác]** giả thuyết ban đầu vì **[...]**.

---

## Phần bonus

Không thực hiện NB3b, NB5, NB6, NB7 hoặc β-sweep trong phạm vi bài nộp bắt buộc này.
