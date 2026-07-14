import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Đảm bảo thư mục gốc dự án được đưa vào sys.path để có thể import các package con ('dataset')
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from dataset import SplitterConfig, FishDataSplitter

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


class FishVoiceDataLoader:
    """
    Lớp duy nhất quản lý toàn bộ quy trình nạp dữ liệu dạng sóng âm thanh của cá (FishVoiceDataLoader).
    Cung cấp đầu vào sóng thô (Raw Waveform) để mô hình thực hiện trích phổ Mel trên GPU.
    Hỗ trợ khởi tạo linh hoạt bằng cách truyền trực tiếp đối tượng cấu hình hoặc truyền các tham số rời rạc.
    """
    # Cờ tĩnh để kiểm soát việc in cảnh báo fallback duy nhất một lần, tránh gây ngập log (log flooding)
    _fallback_warning_shown = False

    def __init__(
        self,
        sample_rate: int = 64000,
        batch_size: int = 40,
        num_workers: int = -1,
        cache_audio: bool = True
    ) -> None:
        """
        Khởi tạo bộ nạp dữ liệu FishVoiceDataLoader.

        Tham số đầu vào (Input):
            sample_rate (int): Tần số lấy mẫu âm thanh. Mặc định: 64000.
            batch_size (int): Kích thước lô dữ liệu. Mặc định: 40.
            num_workers (int): Số luồng CPU song song (-1 để tự động tính toán). Mặc định: -1.
            cache_audio (bool): Cờ bật/tắt cơ chế nạp trước toàn bộ âm thanh vào RAM. Mặc định: True.
        """
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_audio = cache_audio

        # Tự động tính số workers nếu cấu hình là -1 (Tự động phát hiện)
        if self.num_workers == -1:
            max_cpu = os.cpu_count()
            if max_cpu is None or max_cpu <= 0:
                self.num_workers = 0
            elif max_cpu == 2:
                self.num_workers = max_cpu // 2
            else:
                self.num_workers = (max_cpu // 2) + 1

        logger.info("==================================================")
        logger.info("Khởi tạo cấu hình bộ nạp dữ liệu FishVoiceDataLoader:")
        logger.info(f"  - Tần số lấy mẫu (Sample Rate): {self.sample_rate} Hz")
        logger.info(f"  - Kích thước lô (Batch Size):   {self.batch_size}")
        logger.info(f"  - Số luồng CPU (Num Workers):    {self.num_workers} (Tính toán tự động từ {os.cpu_count()} nhân CPU)")
        logger.info(f"  - Nạp trước vào RAM (Cache RAM): {self.cache_audio}")
        logger.info("==================================================")

        # 1. Tải tự động cấu hình bộ chia từ JSON và khởi tạo bộ chia
        self.splitter_config = SplitterConfig.from_json()
        self.splitter = FishDataSplitter(config=self.splitter_config)

        # 2. Thực hiện chia tập dữ liệu và lưu giữ danh sách các tập trong bộ nhớ
        self.train_dict, self.test_dict, self.val_dict = self.splitter.split_data()

        logger.info("Khởi tạo thành công FishVoiceDataLoader.")

    @staticmethod
    def load_audio(path: str, sr: int = 64000) -> torch.Tensor:
        """
        Tải tệp tin âm thanh và resample về tần số lấy mẫu đích.
        """
        waveform, sample_rate = torchaudio.load(path)
        resample_transform = torchaudio.transforms.Resample(sample_rate, sr)
        resample_waveform = resample_transform(waveform)
        return resample_waveform

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gộp nhóm (Collate) các mẫu dạng sóng thô đơn lẻ thành một lô dữ liệu (Batch) 2D.
        Khớp cú pháp 100% với hàm collate_fn của file fish_audio_dataset.py gốc.
        """
        wav_names = [data['audio_name'] for data in batch]
        wav = [data['waveform'] for data in batch]
        target = [data['target'] for data in batch]

        # Khớp tuyệt đối cú pháp gốc của tác giả
        waveforms = torch.FloatTensor(np.array(wav))
        targets_tensor = torch.FloatTensor(np.array(target))

        return {
            'audio_name': wav_names,
            'waveform': waveforms,
            'target': targets_tensor
        }

    class _InnerDataset(Dataset):
        """
        Lớp Dataset nội bộ (nằm trong FishVoiceDataLoader) thực thi giao diện PyTorch Dataset.
        """
        def __init__(self, parent: 'FishVoiceDataLoader', split: str) -> None:
            self.parent = parent
            self.split = split
            self.cache_audio = parent.cache_audio
            self.waveform_cache = None
            self.cache_size_mb = 0.0

            if self.split == 'train':
                self.data_dict = parent.train_dict
            elif self.split == 'test':
                self.data_dict = parent.test_dict
            elif self.split == 'val':
                self.data_dict = parent.val_dict
            else:
                raise ValueError(f"Giá trị split '{self.split}' không hợp lệ. Phải thuộc ['train', 'test', 'val'].")

            # Nếu cờ cache_audio bật, thực hiện nạp trước toàn bộ âm thanh vào RAM
            if self.cache_audio:
                self._preload_audio()

        def _preload_audio(self) -> None:
            """
            Nạp trước toàn bộ sóng âm thanh thô của tập dữ liệu vào RAM song song (ThreadPool) để tăng tốc tối đa.
            Đảm bảo thứ tự của cache khớp hoàn toàn 100% với thứ tự data_dict gốc.
            """
            import concurrent.futures

            logger.info(f"Bắt đầu nạp trước tập dữ liệu '{self.split}' vào RAM...")

            # Hàm phụ tải 1 file âm thanh thô
            def load_single_audio(item: List[Any]) -> np.ndarray:
                wav_name = item[0]
                wav = FishVoiceDataLoader.load_audio(wav_name, sr=self.parent.sample_rate)
                wav_1d = wav.squeeze(0) if wav.ndim > 1 else wav
                return wav_1d.numpy()

            num_threads = self.parent.num_workers
            if num_threads <= 0:
                num_threads = 1

            logger.info(f"Đang chạy tải song song dữ liệu âm thanh với {num_threads} luồng CPU...")

            cache = [None] * len(self.data_dict)
            total_bytes = 0

            # Kiểm tra an toàn thư viện tqdm
            try:
                from tqdm import tqdm
                has_tqdm = True
            except ImportError:
                has_tqdm = False

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {
                    executor.submit(load_single_audio, item): idx 
                    for idx, item in enumerate(self.data_dict)
                }

                if has_tqdm:
                    pbar = tqdm(total=len(self.data_dict), desc=f"Nạp trước tập {self.split} vào RAM")
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        wav_numpy = future.result()
                        cache[idx] = wav_numpy
                        total_bytes += wav_numpy.nbytes
                        pbar.update(1)
                    pbar.close()
                else:
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        wav_numpy = future.result()
                        cache[idx] = wav_numpy
                        total_bytes += wav_numpy.nbytes

            self.waveform_cache = cache
            self.cache_size_mb = total_bytes / (1024 ** 2)
            logger.info(
                f"Đã lưu cache tập {self.split} vào RAM: {len(cache)} tệp âm thanh, {self.cache_size_mb:.1f} MB"
            )

        def __len__(self) -> int:
            return len(self.data_dict)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            item = self.data_dict[index]
            wav_name = item[0]
            target = item[-1]

            # 1. Tải từ cache RAM nếu có, nếu không thì tải động từ đĩa cứng
            if self.waveform_cache is not None:
                wav_numpy = self.waveform_cache[index]
            else:
                wav = FishVoiceDataLoader.load_audio(wav_name, sr=self.parent.sample_rate)
                wav_1d = wav.squeeze(0) if wav.ndim > 1 else wav
                wav_numpy = wav_1d.numpy()

            # 2. Biến đổi nhãn sang One-hot vector 4 lớp (none: 0, strong: 1, medium: 2, weak: 3)
            target_onehot = np.eye(4)[target]

            return {
                'audio_name': wav_name,
                'waveform': wav_numpy,
                'target': target_onehot
            }

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = False,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Khởi tạo và trả về trình nạp dữ liệu DataLoader cho một tập dữ liệu cụ thể.
        """
        dataset = self._InnerDataset(parent=self, split=split)
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )


# Thực thi chính nếu chạy trực tiếp file này (chỉ để tham khảo cách sử dụng)
if __name__ == '__main__':
    # Hướng dẫn sử dụng FishVoiceDataLoader trong mã nguồn Python:
    # 
    # từ dataset.dataloader_melspectrogram import FishVoiceDataLoader
    # 
    # 1. Khởi tạo trực tiếp bằng các tham số rời rạc (Bật cache_audio=True):
    # loader_manager = FishVoiceDataLoader(sample_rate=64000, batch_size=40, num_workers=-1, cache_audio=True)
    # 
    # 2. Tạo các DataLoader tương ứng cho Train/Val/Test:
    # train_loader = loader_manager.get_dataloader(split='train', shuffle=True)
    # val_loader = loader_manager.get_dataloader(split='val', shuffle=False)
    # test_loader = loader_manager.get_dataloader(split='test', shuffle=False)
    # 
    # 3. Sử dụng DataLoader trong vòng lặp huấn luyện:
    # cho batch trong train_loader:
    #     audio_names = batch['audio_name']
    #     waveforms = batch['waveform']  # Tensor dạng sóng thô 2D shape [Batch, Num_Samples]
    #     targets = batch['target']      # Tensor nhãn [Batch, 4] (One-hot FloatTensor)
    # pass
    pass
