import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import L2ArcticDataset, MDDCollate
from src.model.mmd_model_v2 import MDDModelV2

def evaluate_model():
    print("=== BẮT ĐẦU KIỂM THỬ MÔ HÌNH TRÊN TẬP TEST ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. ĐỊNH NGHĨA TẬP TEST KHÁCH QUAN (UNSEEN SPEAKERS)
    root_dir = "./data/raw"
    # Lấy những người nói KHÔNG CÓ trong file train.py
    # Giả sử bạn lấy THV (Tiếng Việt) và HKK (Tiếng Hàn)
    #test_speakers = ["TLV", "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA"] 
    test_speakers = ["suitcase_corpus"] 
    
    print(f"[*] Đang load Test Dataset với các speakers: {test_speakers}...")
    test_dataset = L2ArcticDataset(root_dir=root_dir, speaker_list=test_speakers)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=2, # Cứ để nhỏ để tránh quá tải VRAM
        shuffle=False, # Khi test thì không cần xáo trộn dữ liệu
        collate_fn=MDDCollate(pad_phoneme_id=0),
        num_workers=2
    )

    # 2. LOAD MÔ HÌNH TỐT NHẤT VỪA TRAIN
    print("[*] Đang load trọng số best_mdd_model_v3.pt...")
    vocab_size = 46 
    model = MDDModelV2(vocab_size=vocab_size).to(device)
    
    model_path = "./checkpoints/best_mdd_model_v3.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # ĐẶC BIỆT QUAN TRỌNG: Chuyển mô hình sang chế độ đánh giá (tắt Dropout, Batch Norm)
    model.eval()

    # 3. QUY TRÌNH QUÉT DỮ LIỆU VÀ CHẤM ĐIỂM
    total_true_errors = 0
    total_false_errors = 0
    total_actual_errors = 0
    total_correct_preds = 0
    total_valid_phonemes = 0

    progress_bar = tqdm(test_loader, desc="Đang chạy kiểm thử")
    
    # Tắt tính toán đạo hàm để tăng tốc độ và tiết kiệm bộ nhớ
    with torch.no_grad():
        for batch in progress_bar:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            canonical_ids = batch["canonical_ids"].to(device)
            target_scores = batch["target_scores"].to(device)
            
            logits, _ = model(
                input_values=input_values, 
                attention_mask=attention_mask, 
                canonical_ids=canonical_ids
            )
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            # Chỉ tính toán trên các âm vị hợp lệ (loại bỏ PAD -100.0)
            valid_indices = (target_scores != -100.0)
            preds_valid = preds[valid_indices]
            targets_valid = target_scores[valid_indices]
            
            # Thống kê cho lớp LỖI (0.0)
            true_errors = ((preds_valid == 0.0) & (targets_valid == 0.0)).sum().item()
            false_errors = ((preds_valid == 0.0) & (targets_valid == 1.0)).sum().item()
            actual_errors = (targets_valid == 0.0).sum().item()
            
            # Tổng số âm vị đúng/sai chung
            correct_preds = (preds_valid == targets_valid).sum().item()
            
            total_true_errors += true_errors
            total_false_errors += false_errors
            total_actual_errors += actual_errors
            total_correct_preds += correct_preds
            total_valid_phonemes += valid_indices.sum().item()

    # 4. TỔNG HỢP VÀ IN BÁO CÁO KẾT QUẢ
    accuracy = total_correct_preds / (total_valid_phonemes + 1e-8)
    precision = total_true_errors / (total_true_errors + total_false_errors + 1e-8)
    recall = total_true_errors / (total_actual_errors + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    print("\n" + "="*50)
    print("BÁO CÁO KIỂM THỬ (TEST SET EVALUATION)")
    print("="*50)
    print(f"Tổng số âm vị đã kiểm tra : {total_valid_phonemes}")
    print(f"Tổng số lỗi thực tế có    : {total_actual_errors}")
    print(f"Số lỗi mô hình bắt trúng  : {total_true_errors}")
    print("-" * 50)
    print(f"Độ chính xác tổng (Accuracy) : {accuracy:.4f} (Lưu ý: Chỉ số này thường ảo do mất cân bằng dữ liệu)")
    print(f"Độ chuẩn xác báo lỗi (Precision) : {precision:.4f} (Khả năng tránh báo động giả)")
    print(f"Độ bao phủ lỗi (Recall)      : {recall:.4f} (Khả năng không bỏ lọt lỗi)")
    print(f"Điểm F1-Score (Lớp Lỗi)      : {f1_score:.4f} <--- CHỈ SỐ QUAN TRỌNG NHẤT")
    print("="*50)