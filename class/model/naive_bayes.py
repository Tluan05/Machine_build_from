import numpy as np



class NaiveBayes:
    def fit(self, X, y):
     
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        
        so_mau, so_dac_trung = X.shape
        self._cac_nhan = np.unique(y)         
        so_lop = len(self._cac_nhan)

        
        self._tong_tu_theo_lop = np.zeros((so_lop, so_dac_trung), dtype=np.float64)
        # _so_mau_theo_lop[i] = số lượng mẫu thuộc lớp i
        self._so_mau_theo_lop = np.zeros(so_lop, dtype=np.float64)

        #  Đếm tần suất từng đặc trưng trong mỗi lớp ===
        for i, lop in enumerate(self._cac_nhan):
            X_lop = X[y == lop]  # Lấy tất cả các mẫu thuộc lớp lop
            self._tong_tu_theo_lop[i, :] = X_lop.sum(axis=0)  # Tổng tần suất từng đặc trưng
            self._so_mau_theo_lop[i] = X_lop.shape[0]         # Số mẫu trong lớp này

        #  Thêm Laplace smoothing để tránh chia cho 0 ===
        self._tong_tu_theo_lop += 1

        # Tính log-xác suất P(x_j | lớp) cho từng đặc trưng 
        # Công thức: P(x_j | lớp_i) = (count_ij + 1) / (tổng_từ_lớp_i + số_đặc_trưng)
        self._xac_suat_dac_trung_log = np.log(
            self._tong_tu_theo_lop / self._tong_tu_theo_lop.sum(axis=1, keepdims=True)
        )

        # === Bước 4: Tính log-xác suất tiên nghiệm P(lớp_i) ===
        self._xac_suat_lop_log = np.log(self._so_mau_theo_lop / so_mau)

    def _du_doan_mau(self, x):
     
        if not isinstance(x, np.ndarray):
            x = x.toarray().ravel()

        # Tính log(P(lớp)) + sum(x_i * log(P(x_i | lớp)))
        log_xac_suat_theo_lop = []
        for i, lop in enumerate(self._cac_nhan):
            # Phần log-likelihood: tổng trọng số đặc trưng * log xác suất đặc trưng
            log_likelihood = np.sum(x * self._xac_suat_dac_trung_log[i])
            # Cộng thêm log prior P(lớp)
            log_tong = self._xac_suat_lop_log[i] + log_likelihood
            log_xac_suat_theo_lop.append(log_tong)

        # Lấy lớp có log-xác suất lớn nhất
        return self._cac_nhan[np.argmax(log_xac_suat_theo_lop)]

    def predict(self, X):
       
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Dự đoán từng mẫu
        du_doan = [self._du_doan_mau(x) for x in X]
        return np.array(du_doan)


