import os
import sys
import glob
import logging
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
from itertools import chain
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

# Cấu hình buộc stdout/stderr dùng mã hóa UTF-8 để không bị UnicodeEncodeError trên terminal Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Cấu hình logging của Python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SplitterConfig(BaseModel):
    """
    Lớp cấu hình sử dụng Pydantic để kiểm tra và xác thực dữ liệu đầu vào.
    
    Các tham số đầu vào (Input Parameters):
        dataset_path (str): Đường dẫn tuyệt đối đến thư mục chứa dữ liệu gốc của U_FFIA.
            Ví dụ: 'C:/raw_dataset/U_FFIA'
        seed (int): Số nguyên dùng làm seed ngẫu nhiên để đảm bảo phân chia có thể tái lập.
            Mặc định: 25. Ràng buộc: >= 0.
        test_sample_per_class (int): Số lượng mẫu quy định cho mỗi class trong tập Test và tập Val.
            Mặc định: 700. Ràng buộc: > 0.
        save_results (bool): Biến cờ bật/tắt tự động lưu kết quả ra file.
            Mặc định: True.
    """
    dataset_path: str = Field(
        default='C:/Users/manhm/Desktop/Capstone_2026_Fish_Feeding_Intensity_DL/raw_dataset/U_FFIA',
        description="Đường dẫn tuyệt đối đến thư mục chứa dữ liệu U_FFIA."
    )
    seed: int = Field(
        default=25,
        ge=0,
        description="Seed ngẫu nhiên để đảm bảo khả năng tái lặp (phải >= 0)."
    )
    test_sample_per_class: int = Field(
        default=700,
        gt=0,
        description="Số lượng mẫu cho tập Test/Val trên mỗi class (phải > 0)."
    )
    save_results: bool = Field(
        default=True,
        description="Quyết định tự động lưu kết quả ra file CSV, JSONL và JSON hay không."
    )


class BaseDataSplitter(ABC):
    """
    Lớp cơ sở trừu tượng định nghĩa giao diện chia dataset và các tiện ích dùng chung.
    Xử lý logic xuất file phân chia dùng chung (CSV, JSONL, summary JSON) cho các bộ chia.
    """
    def __init__(self, config: SplitterConfig) -> None:
        self.config = config
        self.dataset_path = config.dataset_path
        self.seed = config.seed
        self.test_sample_per_class = config.test_sample_per_class
        self.save_results = config.save_results

        # Tự động xác định đường dẫn thư mục audio (được dùng làm gốc phân chia mẫu)
        audio_dir = os.path.join(self.dataset_path, 'audio')
        if os.path.isdir(audio_dir):
            self.audio_path = audio_dir
        else:
            self.audio_path = self.dataset_path

    @abstractmethod
    def get_file_list(self, split_name: str) -> List[str]:
        """
        Quét thư mục và trả về danh sách các đường dẫn file.
        Bắt buộc các lớp con phải tự thực thi.

        Đầu vào (Input):
            split_name (str): Tên của lớp cường độ cho ăn cần quét.
                Ví dụ: 'strong', 'medium', 'weak', 'none'

        Đầu ra (Output):
            List[str]: Danh sách chứa đường dẫn tuyệt đối của tất cả các file quét được.
        """
        pass

    @abstractmethod
    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        """
        Thực hiện phân chia danh sách file thành các tập Train, Test và Validation.
        Bắt buộc các lớp con phải tự thực thi.

        Đầu vào (Input):
            Không có tham số (Thông số lấy trực tiếp từ các thuộc tính lớp).

        Đầu ra (Output):
            Tuple[List[List], List[List], List[List]]: Bộ ba chứa lần lượt 3 danh sách (Train, Test, Val).
        """
        pass

    @abstractmethod
    def _format_samples(self, split_name: str, data_list: List[List]) -> List[Dict[str, Any]]:
        """
        Định dạng dữ liệu danh sách thành cấu trúc dictionary chuẩn để xuất file.
        Bắt buộc các lớp con phải tự thực thi.

        Đầu vào (Input):
            split_name (str): Tên tập dữ liệu ('train', 'test', hoặc 'val').
            data_list (List[List]): Danh sách các cặp dữ liệu ghép nhãn dạng [đường_dẫn, nhãn_số].
        
        Đầu ra (Output):
            List[Dict[str, Any]]: Danh sách các dictionary chứa metadata chi tiết của mỗi mẫu.
        """
        pass

    @staticmethod
    def _write_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
        """Ghi danh sách mẫu dữ liệu ra file định dạng JSONL."""
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_csv(samples: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
        """Ghi danh sách mẫu dữ liệu ra file định dạng CSV."""
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

    def _save_splits(self, train_dict: List[List], test_dict: List[List], val_dict: List[List], output_dir: Path) -> None:
        """
        Lưu các tập phân chia ra các file CSV, JSONL và file tóm tắt JSON (Thực thi dùng chung).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        splits = {
            "train": train_dict,
            "test": test_dict,
            "val": val_dict
        }

        formatted_splits: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Định dạng dữ liệu và lưu CSV/JSONL cho từng tập bằng thực thi của lớp con
        for split_name, data_list in splits.items():
            samples = self._format_samples(split_name, data_list)
            formatted_splits[split_name] = samples

            # Tiến hành ghi file
            fieldnames = ["video_path", "audio_path", "label", "class_name", "date", "session", "sample_id"]
            self._write_jsonl(samples, output_path / f"{split_name}.jsonl")
            self._write_csv(samples, output_path / f"{split_name}.csv", fieldnames=fieldnames)
            logger.info(f"Đã lưu thành công các file phân chia vào {output_path}")

        # 2. Tạo và lưu file summary.json
        summary: Dict[str, Any] = {
            "label_map": {
                "none": 0,
                "strong": 1,
                "medium": 2,
                "weak": 3
            },
            "splits": {}
        }
        
        for split_name, samples in formatted_splits.items():
            counts = {"none": 0, "strong": 0, "medium": 0, "weak": 0}
            for s in samples:
                counts[str(s["class_name"])] += 1
            summary["splits"][split_name] = {
                "total": len(samples),
                "by_class": counts
            }

        # Bổ sung siêu dữ liệu cấu hình
        summary.update({
            "dataset_root": self.dataset_path,
            "output_dir": str(output_path),
            "seed": self.seed,
            "test_sample_per_class": self.test_sample_per_class
        })

        with (output_path / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu thành công file tóm tắt phân chia vào {output_path}")


class FishDataSplitter(BaseDataSplitter):
    """
    Lớp thực thi phân chia dữ liệu đa phương thức của cá (Âm thanh + Video).
    Giữ nguyên logic quét file và chia tập từ fish_audio_dataset.py của U-FFIA.
    Tự động liên kết video với audio và fallback sửa đường dẫn nếu copy từ máy khác qua.
    """
    # noinspection DuplicatedCode
    def get_file_list(self, split_name: str) -> List[str]:
        # Giữ nguyên logic hàm get_wav_name trong mã nguồn U-FFIA để quét dữ liệu gốc
        path = self.audio_path
        audio = []
        l1 = os.listdir(path)
        for folder_name in l1:
            l2 = os.listdir(os.path.join(path, folder_name))
            for session_folder in l2:
                wav_dir = os.path.join(path, folder_name, session_folder, split_name, '*.wav')
                audio.append(glob.glob(wav_dir))
        return list(chain.from_iterable(audio))

    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        # Tự động xác định thư mục splits:
        # Nếu dataset_path trỏ vào thư mục audio/video con, thư mục splits sẽ nằm song song với nó.
        # Nếu dataset_path trỏ thẳng vào thư mục gốc U_FFIA, thư mục splits sẽ nằm ngay bên trong U_FFIA.
        if Path(self.dataset_path).name in ['audio', 'video']:
            splits_dir = Path(self.dataset_path).parent / 'splits'
        else:
            splits_dir = Path(self.dataset_path) / 'splits'

        train_csv = splits_dir / 'train.csv'
        test_csv = splits_dir / 'test.csv'
        val_csv = splits_dir / 'val.csv'

        if splits_dir.exists() and train_csv.exists() and test_csv.exists() and val_csv.exists():
            logger.info("==================================================")
            logger.info(f"Kích hoạt chế độ Fallback: Phát hiện phân chia có sẵn tại '{splits_dir}'.")
            logger.info("Đang nạp tập dữ liệu đã chia thay vì tính toán lại...")
            logger.info("==================================================")

            train_dict = []
            test_dict = []
            val_dict = []
            need_rewrite_files = False

            try:
                # Hàm helper đọc dữ liệu từ CSV và tự động sửa đường dẫn file kèm cơ chế gom log
                def load_and_fix_csv(csv_path: Path, split_name: str) -> List[List]:
                    nonlocal need_rewrite_files
                    loaded_data = []
                    fixed_count = 0
                    missing_count = 0
                    
                    with csv_path.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            raw_path = row["audio_path"]
                            label = int(row["label"])
                            
                            # Nếu đường dẫn audio không tồn tại, tự động sửa đổi
                            if not os.path.exists(raw_path):
                                parts = Path(raw_path).parts
                                if len(parts) >= 4:
                                    # Khôi phục đường dẫn audio tuyệt đối dựa trên thư mục hiện tại
                                    fixed_path = os.path.join(self.audio_path, parts[-4], parts[-3], parts[-2], parts[-1])
                                    if os.path.exists(fixed_path):
                                        raw_path = fixed_path
                                        fixed_count += 1
                                        need_rewrite_files = True
                                    else:
                                        missing_count += 1
                            loaded_data.append([raw_path, label])
                    
                    # In log gộp thay vì in tràn lan cho từng file một
                    if fixed_count > 0:
                        logger.info(f"Tập {split_name}: Tự động phát hiện và sửa đổi thành công {fixed_count} đường dẫn sai lệch.")
                    if missing_count > 0:
                        logger.warning(f"Tập {split_name}: Có {missing_count} tệp tin không tồn tại trên máy hiện tại (vui lòng kiểm tra lại bộ dữ liệu).")
                        
                    return loaded_data

                train_dict = load_and_fix_csv(train_csv, "Train")
                test_dict = load_and_fix_csv(test_csv, "Test")
                val_dict = load_and_fix_csv(val_csv, "Validation")

                logger.info(f"Đã nạp thành công các phân chia từ file:")
                logger.info(f"- Tổng số mẫu tập Train: {len(train_dict)}")
                logger.info(f"- Tổng số mẫu tập Test:  {len(test_dict)}")
                logger.info(f"- Tổng số mẫu tập Val:   {len(val_dict)}")
                
                # Nếu có bất kỳ đường dẫn nào bị sửa đổi, tiến hành ghi đè lại file trên đĩa để đồng bộ
                if need_rewrite_files:
                    logger.info("Đang tự động cập nhật và ghi đè lại các tệp tin phân chia trên đĩa để đồng bộ đường dẫn mới...")
                    self._save_splits(train_dict, test_dict, val_dict, splits_dir)
                    logger.info("Đồng bộ và sửa đổi tệp tin phân chia trên đĩa hoàn tất!")
                
                logger.info("==================================================")
                return train_dict, test_dict, val_dict
            except Exception as e:
                logger.warning(f"Lỗi nạp hoặc tự động sửa phân chia: {e}. Chuyển sang quy trình phân chia tiêu chuẩn.")

        logger.info("==================================================")
        logger.info("Bắt đầu quá trình phân chia dữ liệu âm thanh và video...")
        logger.info(f"Thư mục gốc bộ dữ liệu: '{self.dataset_path}'")
        logger.info(f"Seed ngẫu nhiên: {self.seed}")
        logger.info(f"Số lượng mẫu Test/Val mỗi class: {self.test_sample_per_class}")
        logger.info(f"Tự động lưu kết quả: {self.save_results}")
        logger.info("==================================================")

        # Quét tệp tin và ghi nhận logs số lượng
        logger.info("Đang quét danh sách file âm thanh cho từng class...")
        strong_list = self.get_file_list(split_name='strong')
        logger.info(f"Class 'strong': Tìm thấy {len(strong_list)} files.")
        
        medium_list = self.get_file_list(split_name='medium')
        logger.info(f"Class 'medium': Found {len(medium_list)} files.")
        
        weak_list = self.get_file_list(split_name='weak')
        logger.info(f"Class 'weak': Tìm thấy {len(weak_list)} files.")
        
        none_list = self.get_file_list(split_name='none')
        logger.info(f"Class 'none': Tìm thấy {len(none_list)} files.")

        # Xáo trộn danh sách độc lập
        logger.info(f"Đang tiến hành xáo trộn (shuffle) danh sách độc lập với seed={self.seed}...")
        random_state = np.random.RandomState(self.seed)
        random_state.shuffle(strong_list)
        random_state.shuffle(medium_list)
        random_state.shuffle(weak_list)
        random_state.shuffle(none_list)

        # Thực hiện phân chia các tập
        logger.info("Đang tiến hành phân chia (slicing) các tập dữ liệu...")
        strong_test = strong_list[:self.test_sample_per_class]
        medium_test = medium_list[:self.test_sample_per_class]
        weak_test = weak_list[:self.test_sample_per_class]
        none_test = none_list[:self.test_sample_per_class]

        strong_val = strong_list[self.test_sample_per_class:2*self.test_sample_per_class]
        medium_val = medium_list[self.test_sample_per_class:2*self.test_sample_per_class]
        weak_val = weak_list[self.test_sample_per_class:2*self.test_sample_per_class]
        none_val = none_list[self.test_sample_per_class:2*self.test_sample_per_class]

        strong_train = strong_list[2*self.test_sample_per_class:]
        medium_train = medium_list[2*self.test_sample_per_class:]
        weak_train = weak_list[2*self.test_sample_per_class:]
        none_train = none_list[2*self.test_sample_per_class:]

        # Cảnh báo nếu không đủ dữ liệu chia
        for class_name, size in [('strong', len(strong_list)), ('medium', len(medium_list)), ('weak', len(weak_list)), ('none', len(none_list))]:
            req_samples = 2 * self.test_sample_per_class
            if size < req_samples:
                logger.warning(f"Class '{class_name}' chỉ có {size} files, trong khi cần tối thiểu {req_samples} samples để chia Test và Val.")

        # Ghi log chi tiết kích thước phân chia
        logger.info(f"Chi tiết phân chia mỗi class:")
        logger.info(f"       - class 'strong': Train={len(strong_train)}, Test={len(strong_test)}, Val={len(strong_val)}")
        logger.info(f"       - class 'medium': Train={len(medium_train)}, Test={len(medium_test)}, Val={len(medium_val)}")
        logger.info(f"       - class 'weak':   Train={len(weak_train)}, Test={len(weak_test)}, Val={len(weak_val)}")
        logger.info(f"       - class 'none':   Train={len(none_train)}, Test={len(none_test)}, Val={len(none_val)}")

        # Ánh xạ nhãn và gộp danh sách bằng list comprehension để tránh cảnh báo trùng lặp code
        logger.info("Đang ánh xạ nhãn số nguyên và tạo danh sách tập dữ liệu...")
        train_dict = (
            [[wav, 1] for wav in strong_train] +
            [[wav, 2] for wav in medium_train] +
            [[wav, 3] for wav in weak_train] +
            [[wav, 0] for wav in none_train]
        )
        test_dict = (
            [[wav, 1] for wav in strong_test] +
            [[wav, 2] for wav in medium_test] +
            [[wav, 3] for wav in weak_test] +
            [[wav, 0] for wav in none_test]
        )
        val_dict = (
            [[wav, 1] for wav in strong_val] +
            [[wav, 2] for wav in medium_val] +
            [[wav, 3] for wav in weak_val] +
            [[wav, 0] for wav in none_val]
        )

        # Xáo trộn lại tập Train
        logger.info("Xáo trộn (shuffle) lần cuối cho tập dữ liệu Train...")
        random_state.shuffle(train_dict)

        logger.info("==================================================")
        logger.info("Phân chia hoàn tất thành công!")
        logger.info(f"- Tổng số mẫu tập Train: {len(train_dict)}")
        logger.info(f"- Tổng số mẫu tập Test:  {len(test_dict)}")
        logger.info(f"- Tổng số mẫu tập Val:   {len(val_dict)}")
        logger.info("==================================================")

        # Tự động lưu kết quả vào thư mục splits song song với dataset_path nếu save_results là True
        if self.save_results:
            self._save_splits(train_dict, test_dict, val_dict, splits_dir)

        return train_dict, test_dict, val_dict

    def _format_samples(self, split_name: str, data_list: List[List]) -> List[Dict[str, Any]]:
        """Phần tách thông tin mẫu thành các dictionary chuẩn để ghi file, tự động liên kết video."""
        label_to_class = {0: "none", 1: "strong", 2: "medium", 3: "weak"}
        samples = []
        for item in data_list:
            wav_path = item[0]
            label = item[1]
            class_name = label_to_class.get(label, "")

            # Phân tách ngày, phiên (session) và sample_id từ đường dẫn tuyệt đối
            parts = Path(wav_path).parts
            date_part = parts[-4] if len(parts) >= 4 else ""
            session_part = parts[-3] if len(parts) >= 3 else ""
            stem = Path(wav_path).stem
            sample_id = stem.split("_audio_")[-1] if "_audio_" in stem else stem

            # Tự động tìm đường dẫn video tương ứng bằng cách đổi cấu trúc:
            # audio/.../*_audio_*.wav -> video/.../*_video_*.mp4
            video_path_str = ""
            parts_list = list(parts)
            if "audio" in parts_list:
                idx = parts_list.index("audio")
                parts_list[idx] = "video"
            
            filename = parts_list[-1]
            if filename.endswith(".wav"):
                filename = filename[:-4] + ".mp4"
            if "_audio_" in filename:
                filename = filename.replace("_audio_", "_video_")
            parts_list[-1] = filename
            
            # Khôi phục thành đường dẫn tuyệt đối trên đĩa
            if parts_list[0].endswith('\\') or parts_list[0].endswith('/'):
                reconstructed_video = parts_list[0] + os.path.join(*parts_list[1:])
            else:
                reconstructed_video = os.path.join(*parts_list)

            # Nếu file video thực sự tồn tại trên máy hiện tại, lấy đường dẫn đó. Ngược lại để trống "".
            if os.path.exists(reconstructed_video):
                video_path_str = reconstructed_video

            samples.append({
                "video_path": video_path_str,
                "audio_path": str(wav_path),
                "label": int(label),
                "class_name": class_name,
                "date": date_part,
                "session": session_part,
                "sample_id": sample_id
            })
        return samples


# Thực thi chính từ dòng lệnh CLI
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phân chia U-FFIA dataset cho audio và xuất ra các file CSV, JSONL, JSON."
    )
    parser.add_argument("--dataset-root", type=str, default='C:/Users/manhm/Desktop/Capstone_2026_Fish_Feeding_Intensity_DL/raw_dataset/U_FFIA')
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save_results",
        help="Không tự động lưu kết quả ra thư mục splits"
    )
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--test-sample-per-class", type=int, default=700)
    
    args = parser.parse_args()
    
    # Khởi tạo cấu hình Pydantic
    splitter_config = SplitterConfig(
        dataset_path=args.dataset_root,
        seed=args.seed,
        test_sample_per_class=args.test_sample_per_class,
        save_results=args.save_results
    )
    
    # Khởi tạo bộ chia dữ liệu
    splitter = FishDataSplitter(config=splitter_config)
    
    # Tiến hành thực hiện phân chia
    train_data, test_data, val_data = splitter.split_data()
