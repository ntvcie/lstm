import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import mysql.connector
from datetime import timedelta
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, LayerNormalization, SimpleRNN
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization as TransformerLayerNorm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Add
)
import tensorflow as tf
from tensorflow.keras.layers import Conv1D
# Giao diện Streamlit
hide_st_style = """
<style>
    .block-container {
        padding-top: 0rem;  /* Giảm khoảng cách phía trên */
        padding-left: 1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton > button {
        width: 100%; /* Chiều rộng bằng 100% của sidebar */
    }
</style>
"""
st.set_page_config(
    page_title="Dự báo với Multi-Step LSTM",
    layout="wide",
    page_icon="✅",
    initial_sidebar_state="expanded",
)

#st.markdown("<h1 style='color:#5192e0;'>💻Phân tích dữ liệu than tự cháy mỏ hầm lò</h1>", unsafe_allow_html=True)
st.markdown(hide_st_style, unsafe_allow_html=True)
# Main content
# Sidebar để nhập thông tin
with st.sidebar.expander("💾 Cài đặt kết nối cơ sở dữ liệu", expanded=False):
    #st.title("⚙️ Cài đặt")
    db_host = st.text_input("Host", value="123.24.206.17")
    db_port = st.text_input("Port", value="3306")
    db_user = st.text_input("User", value="admin")
    db_password = st.text_input("Password", value="elatec123!", type="password")
    db_name = st.text_input("Database", value="thantuchay")

# Hàm chuẩn hóa dữ liệu

scaler_type = st.sidebar.selectbox(
    "🔄 Chọn phương pháp chuẩn hóa",
    ["StandardScaler", "MinMaxScaler", "RobustScaler"],
    help="📌 RobustScaler: Xử lý dữ liệu có ngoại lai tốt.\n📌 MinMaxScaler: Chuẩn hóa về [0,1].\n📌 StandardScaler: Đưa dữ liệu về phân phối chuẩn (mean=0, std=1)."
)
# Tạo sidebar để chọn mô hình
model_type = st.sidebar.selectbox(
    "Chọn mô hình:",
    ["LSTM", "GRU", "SimpleRNN", "Transformer"]
)
with st.sidebar.expander("🏷️ Chọn thông số và xử lý",expanded=True):
    # Sidebar để nhập thông tin
    # selected_parameter = st.selectbox("Chọn thông số để huấn luyện hoặc dự báo",
    #                                   ["NhietDo1Tram1", "NhietDo2Tram1", "NhietDo1Tram2", "NhietDo2Tram2", "Co1Tram1",
    #                                    "Co2Tram1", "Co1Tram2", "Co2Tram2", "Oxy1Tram1", "Oxy2Tram1", "Oxy1Tram2",
    #                                    "Oxy2Tram2"])
    selected_parameter = st.selectbox("Chọn thông số để huấn luyện hoặc dự báo",
                                      ["NhietDo1Tram1", "NhietDo2Tram1", "NhietDo1Tram2", "NhietDo2Tram2"])
    LoadProcessDataButton = st.button(
            "⭕ Tải dữ liệu và xử lý",
            #help=f"📌 Bước 1: Nhấn nút này để tải dữ liệu từ cơ sở dữ liệu và tính toán giá trị trung bình, max, min của tập dữ liệu với thông số {selected_parameter}"
        )
with st.sidebar.expander("🔮 Huấn luyện & Dự báo lớn nhất", expanded=True):
    col1, col2 = st.columns([5, 4])  # Cột 1 rộng gấp 5/4 lần cột 2
    with col1:
        TrainMax = st.button(f"🛠️ Huấn luyện")
    with col2:
        ForcastMax = st.button(f"🚀 Dự báo")

with st.sidebar.expander("🔮 Huấn luyện & Dự báo trung bình",expanded=True):
    col1, col2 = st.columns([5, 4])  # Cột 1 rộng gấp 5/4 lần cột 2
    with col1:
        TrainMean = st.button(f"🏋️️ Huấn luyện")
    with col2:
        ForcastMean = st.button(f"🎯 Dự báo")

with st.sidebar.expander("🔮 Huấn luyện & Dự báo nhỏ nhất", expanded=True):
    col1, col2 = st.columns([5, 4])  # Cột 1 rộng gấp 5/4 lần cột 2
    with col1:
        TrainMin = st.button(f"🔧️ Huấn luyện")
    with col2:
        ForcastMin = st.button(f"📊 Dự báo")

with st.sidebar.expander("ℹ️ Hướng dẫn sử dụng",expanded=False):
    st.markdown("""
        **📌 Hướng dẫn sử dụng:**  
        - **Bước 1:** Chọn thông số để huấn luyện mô hình hoặc để dự báo.  
        - **Bước 2:** Nhấn **"Tải dữ liệu và xử lý"** để tải dữ liệu từ cơ sở dữ liệu và tính toán giá trị **trung bình, max, min** của tập dữ liệu với thông số.  
        - **Bước 3.1:** Nhấn **"Huấn luyện mô hình trung bình"** để huấn luyện mô hình dự báo giá trị trung bình từng giờ trong 8 giờ tiếp theo. (Nếu đã huấn luyện rồi, bỏ qua bước này).  
        - **Bước 3.2:** Nhấn **"Huấn luyện mô hình lớn nhất"** để dự báo giá trị lớn nhất.  
        - **Bước 3.3:** Nhấn **"Huấn luyện mô hình nhỏ nhất"** để dự báo giá trị nhỏ nhất.  
        - **Bước 4.1:** Nhấn **"Dự báo giá trị trung bình"** để dự báo giá trị trung bình từng giờ trong 8 giờ tiếp theo.  
        - **Bước 4.2:** Nhấn **"Dự báo giá trị lớn nhất"** để dự báo giá trị lớn nhất.  
        - **Bước 4.3:** Nhấn **"Dự báo giá trị nhỏ nhất"** để dự báo giá trị nhỏ nhất.  
        """)
# Hàm kết nối đến cơ sở dữ liệu MySQL và đọc dữ liệu
def get_all_data(days_back, selected_columns, host, port, user, password, database):
    try:
        conn = mysql.connector.connect(host=host, port=port, user=user, password=password, database=database)
        unique_columns = list(set(selected_columns))
        columns_str = ', '.join(unique_columns)
        end_date = pd.Timestamp.now().floor('h')
        start_date = end_date - pd.Timedelta(days=days_back)
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        query = f"""
        SELECT {columns_str}
        FROM dulieuhalong
        WHERE date_time BETWEEN '{start_str}' AND '{end_str}'
        ORDER BY date_time ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except mysql.connector.Error as err:
        st.error(f"Lỗi kết nối MySQL: {err}")
        return None

# Hàm tính giá trị lớn nhất và trung bình từng giờ
def calculate_hourly_stats(df, column):
    df['hour'] = df['date_time'].dt.floor('h')
    hourly_max = df.groupby('hour')[column].max().reset_index()
    hourly_avg = df.groupby('hour')[column].mean().reset_index()
    hourly_min = df.groupby('hour')[column].min().reset_index()
    full_hours = pd.date_range(start=df['hour'].min(), end=df['hour'].max(), freq="1h")
    full_hours_df = pd.DataFrame({'hour': full_hours})
    hourly_max = full_hours_df.merge(hourly_max, on='hour', how='left')
    hourly_avg = full_hours_df.merge(hourly_avg, on='hour', how='left')
    hourly_min = full_hours_df.merge(hourly_min, on='hour', how='left')
    max_value = df[column].max()
    mean_value = df[column].mean()
    min_value = df[column].min()
    hourly_max[column] = hourly_max[column].fillna(max_value)
    hourly_avg[column] = hourly_avg[column].fillna(mean_value)
    hourly_min[column] = hourly_min[column].fillna(min_value)
    return hourly_max, hourly_avg, hourly_min


# --- HÀM CHUẨN HÓA DỮ LIỆU ---
def normalize_data(data, scaler_type):
    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    return data_scaled, scaler

# Hàm tạo chuỗi dữ liệu đầu vào và đầu ra
def create_dataset(data, n_steps_in, n_steps_out):
    """
    Tạo dữ liệu huấn luyện bằng kỹ thuật Sliding Window.

    :param data: Mảng dữ liệu 1D hoặc 2D (cột thời gian).
    :param n_steps_in: Số bước đầu vào.
    :param n_steps_out: Số bước dự báo.
    :return: X_train, y_train dạng numpy arrays.
    """
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:i + n_steps_in])
        y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out])
    return np.array(X), np.array(y)
#Bạn có thể thêm nhiễu vào X_train trước khi đưa vào mô hình bằng cách sử dụng numpy:
#Chỉ thêm nhiễu vào đầu vào X_train, không thêm vào y_train vì nó sẽ làm sai lệch nhãn đầu ra.
#Bạn có thể thêm nhiễu Gaussian vào dữ liệu đầu vào (X) để làm mô hình tổng quát hóa tốt hơn.
def add_gaussian_noise(X, mean=0, std=0.01):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise
# Hàm kiểm tra tính dừng của dữ liệu
def make_stationary(data):
    stationary_data = np.diff(data, axis=0)
    return stationary_data

# Hàm xây dựng và huấn luyện mô hình LSTM
from tensorflow.keras.callbacks import Callback
# --- Streamlit App ---
#st.title("Biểu đồ hội tụ của mô hình LSTM")
# Placeholder để cập nhật biểu đồ động
loss_chart = st.empty()

class StreamlitPlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.learning_rates = []
        self.lr_chart = st.empty()
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.losses.append(logs['loss'])
        # Giảm kích thước hình vẽ
        fig, ax = plt.subplots(figsize=(3.5, 1.5))  # Điều chỉnh kích thước (rộng 4 inch, cao 3 inch)
        ax.plot(self.losses, label="MSE", color="blue")
        #ax.plot(self.losses, color="blue")

        # Cố định phạm vi trục hoành (x-axis) từ 0 đến 200
        ax.set_xlim(0, 500)
        # Đặt phạm vi trục tung (y-axis) bằng gấp 2 lần giá trị loss lớn nhất
        max_loss = max(self.losses) if self.losses else 1  # Tránh lỗi nếu losses rỗng
        ax.set_ylim(0, max_loss * 1.2)  # Gấp 1.2 lần giá trị loss lớn nhất

        ax.set_xlabel("Số lần lặp (Epoch)")
        ax.set_ylabel("Sai số (MSE)")
        ax.legend()
        # Cập nhật biểu đồ trong Streamlit
        loss_chart.pyplot(fig)
        #================================================================================
        lr = float(self.model.optimizer.lr.numpy())
        self.learning_rates.append(lr)

        fig, ax = plt.subplots(figsize=(3.5, 1.5))
        ax.plot(self.learning_rates, label="LR", color="purple")

        # Cố định phạm vi trục hoành (x-axis) từ 0 đến 200
        ax.set_xlim(0, 500)
        # Đặt phạm vi trục tung (y-axis) bằng gấp 2 lần giá trị loss lớn nhất
        max_learning_rates = max(self.learning_rates) if self.learning_rates else 1  # Tránh lỗi nếu losses rỗng
        ax.set_ylim(0, max_learning_rates * 2)  # Gấp 2 lần giá trị loss lớn nhất

        ax.set_xlabel("Số lần lặp (Epoch)")
        ax.set_ylabel("Tốc độ học (LR)")

        ax.legend()

        self.lr_chart.pyplot(fig)

# Callback tùy chỉnh để in loss sau mỗi 5 epoch
class PrintLossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:  # In loss sau mỗi 5 epoch
            #st.write(f"Epoch {epoch + 1}: Loss = {logs['loss']:.4f}")
            st.write(f"**Sai số bình phương trung bình giữa các giá trị được dự đoán và thực tế MSE**: {logs['loss']:.4f}")

# Callback tùy chỉnh để lưu và in loss cuối cùng
class FinalLossCallback(Callback):
    def on_train_end(self, logs=None):
        # Lấy giá trị loss cuối cùng từ logs
        final_loss = logs.get('loss')
        if final_loss is not None:
            # Hiển thị MSE
            st.subheader("🚀 Đánh giá mô hình")
            st.write(f"✅️ **Sai số bình phương trung bình giữa các giá trị được dự đoán và thực tế MSE**: {final_loss:.4f}")
#thay đổi batch size động trong quá trình huấn luyện nếu mô hình bị mắc kẹt ở cực tiểu cục bộ:
#Giảm batch size khi loss bị kẹt
import tensorflow as tf

class DynamicBatchCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, min_bs=16, max_bs=128, patience=5):
        super().__init__()
        self.dataset = dataset.unbatch()  # Đảm bảo dataset không bị batch sẵn
        self.min_bs = min_bs
        self.max_bs = max_bs
        self.patience = patience
        self.wait = 0
        self.prev_loss = float("inf")

        # Tạo danh sách các batch size hợp lệ (lũy thừa của 2)
        self.valid_batch_sizes = [2 ** x for x in range(3, 8)]  # [-2, -4, 8, 16, 32, 64, 128]
        self.current_bs_index = self.valid_batch_sizes.index(min(self.valid_batch_sizes))  # Bắt đầu từ min_bs

        # Khởi tạo batch size ban đầu
        self.current_bs = self.valid_batch_sizes[self.current_bs_index]
        self.dataset = self.dataset.batch(self.current_bs)

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")

        if loss is None:
            return

        # Điều chỉnh batch size dựa trên loss
        if loss > self.prev_loss:  # Nếu loss tăng, giảm batch size
            if self.current_bs_index > 0:  # Chỉ giảm nếu không phải batch size nhỏ nhất
                self.current_bs_index -= 1
                new_bs = self.valid_batch_sizes[self.current_bs_index]
                print(f"⚠️ Giảm batch size xuống: {new_bs}")
                print(f"⚠️ Mục đích: Giảm batch size giúp mô hình thoát khỏi cực tiểu cục bộ.")
                self.dataset = self.dataset.unbatch().batch(new_bs)
                self.wait = 0
        else:  # Nếu loss giảm, tăng batch size
            self.wait += 1
            if self.wait >= self.patience:
                if self.current_bs_index < len(
                        self.valid_batch_sizes) - 1:  # Chỉ tăng nếu không phải batch size lớn nhất
                    self.current_bs_index += 1
                    new_bs = self.valid_batch_sizes[self.current_bs_index]
                    print(f"✅ Tăng batch size lên: {new_bs}")
                    print(f"✅ Mục đích: Batch size lớn hơn giúp mô hình học nhanh hơn và ổn định hơn.")
                    self.dataset = self.dataset.unbatch().batch(new_bs)
                self.wait = 0

        self.prev_loss = loss



# class StreamlitPlotCallback(Callback):
#     def __init__(self):
#         super().__init__()
#         self.losses = []
#
#     def on_epoch_end(self, epoch, logs=None):
#         if logs is not None:
#             self.losses.append(logs['loss'])
#
#         # Vẽ lại biểu đồ loss
#         fig, ax = plt.subplots()
#         ax.plot(self.losses, label="Loss trên tập huấn luyện", color="blue")
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Loss (MSE)")
#         ax.legend()
#
#         # Cập nhật biểu đồ trong Streamlit
#         loss_chart.pyplot(fig)

# Định nghĩa Positional Encoding
def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

# Định nghĩa khối Transformer
def transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    # MultiHeadAttention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x1 = Add()([x, attention_output])  # Residual connection
    x1 = LayerNormalization()(x1)

    # Feed-Forward Network
    ffn = Dense(ff_dim, activation="relu")(x1)
    ffn = Dense(key_dim)(ffn)
    x2 = Add()([x1, ffn])  # Residual connection
    x2 = LayerNormalization()(x2)

    return x2

def build_transformer_model(n_steps_in, n_steps_out, num_heads=4, d_model=64, ff_dim=128, num_blocks=4, dropout_rate=0.1): # Thêm d_model vào tham số
    inputs = Input(shape=(n_steps_in, 1))

    # === THAY ĐỔI: Chiếu input lên d_model ===
    x = Dense(d_model, activation='relu')(inputs) # Chiếu input (..., 1) -> (..., d_model)
    x = LayerNormalization()(x) # Có thể norm sau khi chiếu

    # === THAY ĐỔI: Tính PE với d_model ===
    pos_encoding = positional_encoding(n_steps_in, d_model)
    x = x + pos_encoding
    x = Dropout(dropout_rate)(x) # Thêm dropout sau PE

    # Thêm CNN (Tùy chọn)
    # x = Conv1D(filters=d_model, kernel_size=3, activation="relu", padding="same")(x)
    # x = LayerNormalization()(x) # Norm sau Conv

    # Khối Transformer
    for _ in range(num_blocks):
        # Truyền d_model vào key_dim hoặc giữ key_dim riêng biệt nếu muốn
        x = transformer_block(x, num_heads=num_heads, key_dim=d_model, ff_dim=ff_dim, dropout_rate=dropout_rate)

    # Fully connected layers
    x = GlobalAveragePooling1D()(x)
    # x = Flatten()(x) # Thử Flatten thay vì GAP
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)

    # Đầu ra
    outputs = Dense(n_steps_out)(x)

    # Xây dựng mô hình
    model = Model(inputs, outputs)
    return model

def train_lstm_model(X_train, y_train, n_steps_in, n_steps_out):
    print("X_train shape ((Số mẫu, 48, 1)):", X_train.shape)
    print("y_train shape ((Số mẫu, 8, 1)):", y_train.shape)
    print("X_train mean:", np.mean(X_train), "std:", np.std(X_train))
    print("y_train mean:", np.mean(y_train), "std:", np.std(y_train))
    print("======================================================Giải thích============================================================")
    print("Giả sử bạn có 280 mẫu, batch_size=32")
    print("280 : 32 ≈ 8.75 batch → Do số batch phải là số nguyên, Keras sẽ tự động lấy đủ 8 batch (256 mẫu), batch cuối chỉ có 24 mẫu.")
    print("Vậy mỗi epoch sẽ chia dữ liệu thành 9 batch: 8 batch đầu có 32 mẫu, 1 batch cuối chỉ có 24 mẫu")
    print("Sau 1 epoch, mô hình đã duyệt qua toàn bộ 280 mẫu.")
    print("Batch 1: 32 mẫu đầu tiên → Tính gradient → Cập nhật trọng số")
    print("Batch 2: 32 mẫu tiếp theo → Tính gradient → Cập nhật trọng số")
    print("...")
    print("Batch 9: 24 mẫu cuối cùng (vì 280 không chia hết cho 32)")
    print("💡 Mỗi epoch vẫn duyệt qua toàn bộ dữ liệu, nhưng theo từng phần nhỏ.")
    print("============================================================================================================================")
    # model = Sequential([
    #     GRU(512, return_sequences=True, input_shape=(n_steps_in, 1)),
    #     BatchNormalization(),  # Thay thế BatchNormalization bằng LayerNormalization
    #     Dropout(0.1),  # Tăng tỷ lệ Dropout
    #     GRU(256, return_sequences=True),  # Thêm return_sequences=True
    #     BatchNormalization(),
    #     Dropout(0.1),
    #     GRU(128),  # Tầng cuối cùng không cần return_sequences=True
    #     BatchNormalization(),
    #     Dropout(0.1),
    #     Dense(n_steps_out)  # Đảm bảo n_steps_out phù hợp với bài toán
    # ])
    if model_type == "Transformer":
        # === THAY ĐỔI: Gọi hàm build Transformer Model ===
        d_model_transformer = 64  # Chọn embedding dimension
        model = build_transformer_model(
            n_steps_in, n_steps_out,
            num_heads=4,
            d_model=d_model_transformer,
            ff_dim=128,
            num_blocks=2,  # === THỬ GIẢM SỐ BLOCK ===
            dropout_rate=0.1
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Giữ Adam để đơn giản
        model.compile(optimizer=optimizer, loss='mse')

    else:
        # Các mô hình khác (LSTM, GRU, SimpleRNN) vẫn sử dụng Sequential
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, GRU, SimpleRNN

        if model_type == "LSTM":
            model = Sequential([
                LSTM(512, activation='tanh', return_sequences=True, input_shape=(n_steps_in, 1)),
                BatchNormalization(),  # Thay thế Batch Normalization bằng Layer Normalization
                Dropout(0.1),
                LSTM(256, activation='tanh', return_sequences=True),
                BatchNormalization(),  # Thay thế Batch Normalization bằng Layer Normalization
                Dropout(0.1),
                LSTM(128, activation='tanh'),
                BatchNormalization(),  # Thay thế Batch Normalization bằng Layer Normalization
                Dropout(0.1),
                Dense(n_steps_out)
                #relu
            ])

        elif model_type == "GRU":
            model = Sequential([
                GRU(512, return_sequences=True, input_shape=(n_steps_in, 1)),
                BatchNormalization(),  # Thay thế BatchNormalization bằng LayerNormalization
                Dropout(0.1),  # Tăng tỷ lệ Dropout
                GRU(256, return_sequences=True),  # Thêm return_sequences=True
                BatchNormalization(),
                Dropout(0.1),
                GRU(128),  # Tầng cuối cùng không cần return_sequences=True
                BatchNormalization(),
                Dropout(0.1),
                Dense(n_steps_out)  # Đảm bảo n_steps_out phù hợp với bài toán
            ])
        elif model_type == "SimpleRNN":
            model = Sequential([
                SimpleRNN(200, return_sequences=True, input_shape=(n_steps_in, 1)),
                Dropout(0.1),
                SimpleRNN(200, return_sequences=False),
                Dropout(0.1),
                Dense(n_steps_out)  # Đảm bảo n_steps_out phù hợp với bài toán
            ])
        model.compile(optimizer='adam', loss='mse')

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="loss",
        patience=25,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.05, # Số nhỏ thif giảm tốc độ học nhẹ hơn
        patience=25,
        verbose=1,
        min_lr=1e-6
    )

    lr_plot_callback = StreamlitPlotCallback()
    #Giảm batch size khi loss bị kẹt
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(64)


    # Huấn luyện mô hình
    model.fit(
        #X_train, y_train,
        train_dataset,
        epochs=500,  # Tăng số epoch
        #batch_size=32,  # Tăng batch size
        verbose=0,
        callbacks=[lr_plot_callback, early_stopping, reduce_lr, DynamicBatchCallback(train_dataset)]
    )

    return model
#==============================Ok===================================================
# def train_lstm_model(X_train, y_train, n_steps_in, n_steps_out):
#     model = Sequential([
#         LSTM(256, activation='relu', return_sequences=True, input_shape=(n_steps_in, 1)),
#         BatchNormalization(),  # Thay thế Batch Normalization bằng Layer Normalization
#         #Dropout(0.1),
#         LSTM(128, activation='relu', return_sequences=True),
#         BatchNormalization(),  # Thay thế Batch Normalization bằng Layer Normalization
#         #Dropout(0.1),
#         LSTM(64, activation='relu'),
#         BatchNormalization(),  # Thay thế Batch Normalization bằng Layer Normalization
#         #Dropout(0.1),
#         Dense(n_steps_out)
#     ])
#     # Chuẩn hóa gradient bằng cách giới hạn giá trị gradient
#     #optimizer = Adam(clipvalue=1.0)  # Giới hạn gradient trong khoảng [-1.0, 1.0]
#     model.compile(optimizer='adam', loss='mse')
#
#     early_stopping = EarlyStopping(
#         monitor="loss",
#         patience=50,  # Tăng patience để cho mô hình học lâu hơn
#         restore_best_weights=True,
#         verbose=1
#     )
#
#     reduce_lr = ReduceLROnPlateau(
#         monitor="loss",
#         factor=0.5,
#         patience=20,  # Tăng patience để giảm learning rate chậm hơn
#         verbose=1,
#         min_lr=1e-6
#     )
#
#     lr_plot_callback = StreamlitPlotCallback()
#
#     model.fit(
#         X_train, y_train,
#         epochs=200,
#         verbose=0,
#         callbacks=[lr_plot_callback, early_stopping, reduce_lr]
#     )
#
#     return model
#=========================================Ok==========================================================
# def train_lstm_model(X_train, y_train, n_steps_in, n_steps_out):
#     # Xây dựng mô hình LSTM
#     model = Sequential([
#         LSTM(300, activation='relu', return_sequences=True, input_shape=(n_steps_in, 1)),
#         LSTM(300, activation='relu'),
#         Dense(n_steps_out)
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     # Callback dừng sớm nếu loss không giảm sau 10 epoch
#     early_stopping = EarlyStopping(
#         monitor="loss",  # Theo dõi loss trên tập huấn luyện
#         patience=20,  # Dừng nếu loss không giảm trong 10 epoch
#         restore_best_weights=True,  # Quay lại trọng số tốt nhất
#         verbose=1
#     )
#
#     # Khởi tạo callback để vẽ online
#     plot_callback = StreamlitPlotCallback()
#
#     # Huấn luyện mô hình với callback để vẽ online và callback để dừng luyện nếu sao 10 epoch mà loss không giảm
#     model.fit(
#         X_train, y_train,
#         epochs=200,
#         verbose=0,  # Ẩn log để tránh làm rối UI Streamlit
#         callbacks=[plot_callback, early_stopping, FinalLossCallback()]
#     )
#
#
#     #st.success("Mô hình đã hoàn thành huấn luyện! 🚀")
#     return model
#=================================Ok====================================================
# def train_lstm_model(X_train, y_train, n_steps_in, n_steps_out):
#     # Xây dựng mô hình LSTM
#     model = Sequential()
#
#     model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_steps_in, 1)))
#     model.add(LSTM(200, activation='relu'))
#     model.add(Dense(n_steps_out))
#     model.compile(optimizer='adam', loss='mse')
#
#     # Huấn luyện mô hình và lưu lại history
#     history = model.fit(
#         X_train, y_train,
#         epochs=200,
#         verbose=1
#     )
#
#     # --- Streamlit App ---
#     st.title("Biểu đồ hội tụ của mô hình LSTM")
#
#     # Vẽ biểu đồ loss
#     st.subheader("Loss theo số epoch")
#     fig, ax = plt.subplots()
#     ax.plot(history.history['loss'], label="Loss trên tập huấn luyện", color="blue")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss (MSE)")
#     ax.legend()
#     st.pyplot(fig)
#
#     # Vẽ biểu đồ Learning Rate nếu có
#     if 'lr' in history.history:
#         st.subheader("Learning Rate theo số epoch")
#         fig2, ax2 = plt.subplots()
#         ax2.plot(history.history['lr'], label="Learning Rate", color="red")
#         ax2.set_xlabel("Epoch")
#         ax2.set_ylabel("Learning Rate")
#         ax2.legend()
#         st.pyplot(fig2)
#
#     st.write("Mô hình đã hoàn thành huấn luyện! 🚀")
#     return model
#=============================Goc========================================================
# def train_lstm_model(X_train, y_train, n_steps_in, n_steps_out):
#     # Xây dựng mô hình LSTM
#     model = Sequential([
#         LSTM(300, activation='relu', input_shape=(n_steps_in, 1), return_sequences=True),
#         BatchNormalization(),
#         Dropout(0.01),
#         LSTM(200, activation='relu', return_sequences=True),
#         BatchNormalization(),
#         Dropout(0.01),
#         LSTM(100, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.01),
#         Dense(50, activation='relu'),
#         Dense(n_steps_out)
#     ])
#
#     # Cấu hình optimizer và loss function
#     optimizer = Adam(learning_rate=0.001)  # Khởi tạo với learning rate cao hơn
#
#     model.compile(optimizer=optimizer, loss='mse')
#
#     # Callback: Dừng sớm nếu loss không giảm sau 10 epoch
#     early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
#
#     # Callback: Giảm learning rate nếu loss không cải thiện sau 5 epoch
#     reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
#
#     # Huấn luyện mô hình và lưu lại history
#     history = model.fit(
#         X_train, y_train,
#         epochs=200,
#         batch_size=32,
#         callbacks=[early_stopping, reduce_lr],
#         verbose=1
#     )
#
#     # --- Streamlit App ---
#     st.title("Biểu đồ hội tụ của mô hình LSTM")
#
#     # Vẽ biểu đồ loss
#     st.subheader("Loss theo số epoch")
#     fig, ax = plt.subplots()
#     ax.plot(history.history['loss'], label="Loss trên tập huấn luyện", color="blue")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss (MSE)")
#     ax.legend()
#     st.pyplot(fig)
#
#     # Vẽ biểu đồ Learning Rate nếu có
#     if 'lr' in history.history:
#         st.subheader("Learning Rate theo số epoch")
#         fig2, ax2 = plt.subplots()
#         ax2.plot(history.history['lr'], label="Learning Rate", color="red")
#         ax2.set_xlabel("Epoch")
#         ax2.set_ylabel("Learning Rate")
#         ax2.legend()
#         st.pyplot(fig2)
#
#     st.write("Mô hình đã hoàn thành huấn luyện! 🚀")
#     return model

# Hàm lưu mô hình và scaler kèm thông số
def save_model_with_param(model, scaler, parameter):
    model_filename = f"model_{parameter}.h5"
    scaler_filename = f"scaler_{parameter}.pkl"
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)

# Hàm tải mô hình và scaler dựa trên thông số
def load_model_with_param(parameter):
    model_filename = f"model_{parameter}.h5"
    scaler_filename = f"scaler_{parameter}.pkl"
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        model = load_model(model_filename)
        scaler = joblib.load(scaler_filename)
        return model, scaler
    return None, None

# Hàm điền đầy đủ dữ liệu cho các giờ bị thiếu
def fill_missing_hours(df, column, start_time, end_time):
    full_hours = pd.date_range(start=start_time, end=end_time, freq="1h")
    full_hours_df = pd.DataFrame({'hour': full_hours})
    df['hour'] = df['date_time'].dt.floor('h')
    df_filled = full_hours_df.merge(df, on='hour', how='left')
    mean_value = df[column].mean()
    df_filled[column] = df_filled[column].fillna(mean_value)
    return df_filled

# Nút nhấn để tải dữ liệu
daybackdata = 14
if LoadProcessDataButton:
    with st.spinner("Đang tải dữ liệu và xử lý..."):
        selected_columns = ["date_time", selected_parameter]
        df_raw = get_all_data(daybackdata, selected_columns, db_host, db_port, db_user, db_password, db_name)
        if df_raw is None or df_raw.empty:
            st.error("Không có dữ liệu để xử lý!")
            st.stop()
        # Tính toán thời gian bắt đầu và kết thúc
        end_time = pd.Timestamp.now().floor('h')
        start_time = end_time - pd.Timedelta(days=daybackdata)
        # Điền đầy đủ dữ liệu cho các giờ bị thiếu
        df_raw = fill_missing_hours(df_raw, selected_parameter, start_time, end_time)
        # Tính giá trị trung bình và lớn nhất từng giờ
        hourly_max, hourly_avg, hourly_min = calculate_hourly_stats(df_raw, selected_parameter)
        st.success("Dữ liệu đã được tải và xử lý thành công!")
        # Tạo layout với hai cột
        col1, col2, col3 = st.columns([1, 1, 1])
        # Bảng ở cột bên trái
        with col1:
            st.write("Bảng giá trị lớn nhất từng giờ:")
            #st.write(hourly_max.head(18))
            st.dataframe(hourly_max, height=min(800, 35 * len(hourly_max)))  # Tự động điều chỉnh chiều cao
        with col2:
            st.write(f"Bảng giá trị trung bình từng giờ: {len(hourly_avg)} mẫu.")
            #st.write(hourly_avg.head(18))
            st.dataframe(hourly_avg, height=min(800, 35 * len(hourly_avg)))
        with col3:
            st.write("Bảng giá trị nhỏ nhất từng giờ:")
            #st.write(hourly_min.head(18))
            st.dataframe(hourly_min, height=min(800, 35 * len(hourly_min)))
        st.session_state.hourly_avg = hourly_avg
        st.session_state.hourly_max = hourly_max
        st.session_state.hourly_min = hourly_min

# Huấn luyện mô hình cho giá trị trung bình
if TrainMean:
    if 'hourly_avg' in st.session_state:
        st.empty()
        with st.spinner(f"Đang huấn luyện mô hình cho giá trị trung bình của {selected_parameter}..."):
            data_avg = st.session_state.hourly_avg[selected_parameter].values
            stationary_avg = make_stationary(data_avg)
            stationary_avg_scaled, scaler_avg = normalize_data(stationary_avg,scaler_type)
            n_steps_in, n_steps_out = 48, 8
            X_avg, y_avg = create_dataset(stationary_avg_scaled, n_steps_in, n_steps_out)
            # Thêm nhiễu vào X_max
            if selected_parameter == "NhietDo2Tram2":
                X_avg = add_gaussian_noise(X_avg, std=0.035)
            else:
                X_avg = add_gaussian_noise(X_avg, std=0.01)  # Điều chỉnh std tùy theo thử nghiệm. Nếu mô hình không hội tụ tốt, hãy thử giảm std. std=0.01: Nhiễu nhẹ std=0.05: Trung bình std=0.1: Nhiễu cao (có thể làm ảnh hưởng đến mô hình)
            X_avg = X_avg.reshape((X_avg.shape[0], X_avg.shape[1], 1))
            model_avg = train_lstm_model(X_avg, y_avg, n_steps_in, n_steps_out)
            save_model_with_param(model_avg, scaler_avg, f"avg_{selected_parameter}")
            st.success(f"Mô hình và scaler cho giá trị trung bình của {selected_parameter} đã được huấn luyện và lưu lại!")
    else:
        st.warning("Vui lòng nhấn nút Tải dữ liệu và xử lý dữ liệu trước khi huấn luyện hoặc dự báo.")
# Huấn luyện mô hình cho giá trị lớn nhất
if TrainMax:
    if 'hourly_max' in st.session_state:
        st.empty()
        with st.spinner(f"Đang huấn luyện mô hình cho giá trị lớn nhất của {selected_parameter}..."):
            data_max = st.session_state.hourly_max[selected_parameter].values
            stationary_max = make_stationary(data_max)
            stationary_max_scaled, scaler_max = normalize_data(stationary_max,scaler_type)
            n_steps_in, n_steps_out = 48, 8
            X_max, y_max = create_dataset(stationary_max_scaled, n_steps_in, n_steps_out)
            # Thêm nhiễu vào X_max
            X_max = add_gaussian_noise(X_max, std=0.01)  # Điều chỉnh std tùy theo thử nghiệm. Nếu mô hình không hội tụ tốt, hãy thử giảm std. std=0.01: Nhiễu nhẹ std=0.05: Trung bình std=0.1: Nhiễu cao (có thể làm ảnh hưởng đến mô hình)
            X_max = X_max.reshape((X_max.shape[0], X_max.shape[1], 1))
            model_max = train_lstm_model(X_max, y_max, n_steps_in, n_steps_out)
            save_model_with_param(model_max, scaler_max, f"max_{selected_parameter}")
            st.success(f"Mô hình và scaler cho giá trị lớn nhất của {selected_parameter} đã được huấn luyện và lưu lại!")
    else:
        st.warning("Vui lòng nhấn nút Tải dữ liệu và xử lý dữ liệu trước khi huấn luyện hoặc dự báo.")
# Huấn luyện mô hình cho giá trị nhỏ nhất
if TrainMin:
    if 'hourly_min' in st.session_state:
        st.empty()
        with st.spinner(f"Đang huấn luyện mô hình cho giá trị nhỏ nhất của {selected_parameter}..."):
            data_min = st.session_state.hourly_min[selected_parameter].values
            stationary_min = make_stationary(data_min)
            stationary_min_scaled, scaler_min = normalize_data(stationary_min,scaler_type)
            n_steps_in, n_steps_out = 48, 8
            X_min, y_min = create_dataset(stationary_min_scaled, n_steps_in, n_steps_out)
            X_min = X_min.reshape((X_min.shape[0], X_min.shape[1], 1))
            model_min = train_lstm_model(X_min, y_min, n_steps_in, n_steps_out)
            save_model_with_param(model_min, scaler_min, f"min_{selected_parameter}")
            st.success(f"Mô hình và scaler cho giá trị nhỏ nhất của {selected_parameter} đã được huấn luyện và lưu lại!")
    else:
        st.warning("Vui lòng nhấn nút Tải dữ liệu và xử lý dữ liệu trước khi huấn luyện hoặc dự báo.")
import numpy as np
def handle_outliers(data):
    """
    Hàm kiểm tra và xử lý giá trị ngoại lệ trong mảng dữ liệu.
    - Phát hiện giá trị ngoại lệ sử dụng quy tắc IQR.
    - Thay thế giá trị ngoại lệ bằng giá trị trung bình của dữ liệu không chứa ngoại lệ.

    :param data: Mảng dữ liệu đầu vào (numpy array hoặc list).
    :return: Mảng dữ liệu đã được xử lý.
    """
    # Chuyển đổi dữ liệu thành numpy array nếu chưa phải
    data = np.array(data)

    # Tính toán IQR (Interquartile Range)
    Q1 = np.percentile(data, 25)  # Phần vị 25%
    Q3 = np.percentile(data, 75)  # Phần vị 75%
    IQR = Q3 - Q1  # Khoảng IQR

    # Xác định ngưỡng dưới và ngưỡng trên
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Tìm các giá trị ngoại lệ
    outliers = (data < lower_bound) | (data > upper_bound)

    # Tính giá trị trung bình của dữ liệu không chứa ngoại lệ
    non_outliers = data[~outliers]
    mean_non_outliers = np.mean(non_outliers) if len(non_outliers) > 0 else np.mean(data)

    # Thay thế giá trị ngoại lệ bằng giá trị trung bình
    data[outliers] = mean_non_outliers

    return data

# Dự báo giá trị trung bình
if ForcastMean:
    if 'hourly_avg' in st.session_state:
        st.empty()
        model_avg, scaler_avg = load_model_with_param(f"avg_{selected_parameter}")
        if model_avg is None or scaler_avg is None:
            st.error(f"Không tìm thấy mô hình hoặc scaler cho giá trị trung bình của {selected_parameter}!")
            st.stop()
        with st.spinner(f"Đang dự báo giá trị trung bình của {selected_parameter}..."):
            # Lấy dữ liệu cuối cùng (49 giờ gần nhất)
            last_49_avg = st.session_state.hourly_avg[selected_parameter].values[-49:]

            # Kiểm tra và xử lý giá trị ngoại lệ
            last_49_avg = handle_outliers(last_49_avg)

            original_value = last_49_avg[0]
            last_48_avg_diff = make_stationary(last_49_avg)
            last_48_avg_scaled, _ = normalize_data(last_48_avg_diff,scaler_type)
            if last_48_avg_scaled.shape[0] < 48:
                st.error(f"Không đủ dữ liệu để dự báo! Kích thước dữ liệu: {last_48_avg_scaled.shape[0]} mẫu.")
                st.stop()
            last_48_avg_scaled = last_48_avg_scaled.reshape((1, 48, 1))
            forecast_avg_scaled = model_avg.predict(last_48_avg_scaled)
            forecast_avg_diff = scaler_avg.inverse_transform(forecast_avg_scaled.reshape(-1, 1))
            forecast_avg_real = np.cumsum(forecast_avg_diff.flatten()) + original_value
            last_hour = st.session_state.hourly_avg["hour"].iloc[-1]
            forecast_time = pd.date_range(start=last_hour, periods=9, freq="h")[1:]

            st.title(f"📊 Dự báo giá trị trung bình của {selected_parameter} trong 8 giờ tiếp theo")

            forecast_df = pd.DataFrame({
                "Thời gian": forecast_time,
                "Giá trị dự báo": forecast_avg_real
            })
            # Tổ chức giao diện thành các tab
            tab1, tab2 = st.tabs(["Biểu đồ", "Bảng dữ liệu"])

            with tab1:

                # Vẽ biểu đồ với Plotly
                fig = go.Figure()

                # Giá trị thực tế
                actual_values = st.session_state.hourly_avg[f"{selected_parameter}"].values[-48:]
                actual_time = st.session_state.hourly_avg["hour"].iloc[-48:].values

                # Chuyển đổi actual_time thành datetime
                actual_time = pd.to_datetime(actual_time)

                # Đảm bảo forecast_time bắt đầu ngay sau actual_time[-1]
                if forecast_time[0] != actual_time[-1] + pd.Timedelta(hours=1):
                    forecast_time = pd.date_range(start=actual_time[-1] + pd.Timedelta(hours=1), periods=8, freq="h")

                fig.add_trace(go.Scatter(
                    x=actual_time,
                    y=actual_values,
                    mode='lines+markers',
                    name="Giá trị thực tế",
                    line=dict(color="green", dash="dot"),
                    marker=dict(symbol="x")
                ))

                # Giá trị dự báo
                fig.add_trace(go.Scatter(
                    x=forecast_time,
                    y=forecast_avg_real,
                    mode='lines+markers',
                    name="Dự báo",
                    line=dict(color="red", dash="dot"),
                    marker=dict(symbol="circle")
                ))

                # Nét nối giữa giá trị thực tế và dự báo
                connection_x = [actual_time[-1], forecast_time[0]]
                connection_y = [actual_values[-1], forecast_avg_real[0]]
                fig.add_trace(go.Scatter(
                    x=connection_x,
                    y=connection_y,
                    mode='lines',
                    name="Nối giá trị",
                    line=dict(color="orange", dash="dot")
                ))

                # Cấu hình layout
                fig.update_layout(
                    title="📉 Giá trị trung bình thực tế 48 giờ qua và giá trị dự báo trong 8 giờ tiếp theo",
                    xaxis_title="Thời gian",
                    yaxis_title="Giá trị trung bình",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                st.plotly_chart(fig)

            with tab2:
                # Hiển thị bảng dữ liệu
                st.write(forecast_df)

            # Hiển thị thông tin thống kê
            #st.subheader("📊 Thông tin thống kê dữ liệu dự báo")
            col1, col2, col3, col4, col5 = st.columns(5)

            # 1. Trung bình
            mean_value = np.mean(forecast_avg_real)
            #col1.metric("Trung bình", f"{mean_value:.2f} °C")

            # Thêm biểu đồ gauge cho trung bình
            fig_mean = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mean_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Trung bình"},
                gauge={
                    'axis': {'range': [None, 60]},  # Phạm vi nhiệt độ tối đa
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 50], 'color': "orange"},
                        {'range': [50, 60], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': mean_value
                    }
                }
            ))
            col1.plotly_chart(fig_mean, use_container_width=True)

            # 2. Lớn nhất
            max_value = np.max(forecast_avg_real)
            #col2.metric("Lớn nhất", f"{max_value:.2f} °C")

            # Thêm biểu đồ gauge cho giá trị lớn nhất
            fig_max = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Lớn nhất"},
                gauge={
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 50], 'color': "orange"},
                        {'range': [50, 60], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': max_value
                    }
                }
            ))
            col2.plotly_chart(fig_max, use_container_width=True)

            # 3. Nhỏ nhất
            min_value = np.min(forecast_avg_real)
            #col3.metric("Nhỏ nhất", f"{min_value:.2f} °C")

            # Thêm biểu đồ gauge cho giá trị nhỏ nhất
            fig_min = go.Figure(go.Indicator(
                mode="gauge+number",
                value=min_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Nhỏ nhất"},
                gauge={
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 50], 'color': "orange"},
                        {'range': [50, 60], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': min_value
                    }
                }
            ))
            col3.plotly_chart(fig_min, use_container_width=True)

            # Đánh giá xu hướng nhiệt độ
            temperature_changes = np.diff(forecast_avg_real)  # Sự thay đổi giữa các giờ liên tiếp

            # Tính tỷ lệ tăng/giảm
            positive_changes = np.sum(temperature_changes > 0)  # Số lần tăng
            negative_changes = np.sum(temperature_changes < 0)  # Số lần giảm
            total_changes = len(temperature_changes)

            # Ngưỡng tỷ lệ tăng/giảm (ví dụ: > 60%)
            increase_ratio = positive_changes / total_changes if total_changes > 0 else 0
            decrease_ratio = negative_changes / total_changes if total_changes > 0 else 0

            # Hiển thị tỷ lệ tăng/giảm
            #st.write(f"📊 Tỷ lệ tăng: {increase_ratio * 100:.1f}% | Tỷ lệ giảm: {decrease_ratio * 100:.1f}%")

            # Biểu đồ gauge cho tỷ lệ tăng
            fig_increase = go.Figure(go.Indicator(
                mode="gauge+number",
                value=increase_ratio * 100,  # Chuyển tỷ lệ về phần trăm
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Tỷ lệ tăng (%)"},
                gauge={
                    'axis': {'range': [None, 100]},  # Phạm vi từ 0% đến 100%
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},  # An toàn
                        {'range': [30, 60], 'color': "orange"},  # Cảnh báo
                        {'range': [60, 100], 'color': "red"}  # Nguy hiểm
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': increase_ratio * 100  # Giá trị hiện tại
                    }
                }
            ))
            col4.plotly_chart(fig_increase, use_container_width=True)

            # Biểu đồ gauge cho tỷ lệ giảm
            fig_decrease = go.Figure(go.Indicator(
                mode="gauge+number",
                value=decrease_ratio * 100,  # Chuyển tỷ lệ về phần trăm
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Tỷ lệ giảm (%)"},
                gauge={
                    'axis': {'range': [None, 100]},  # Phạm vi từ 0% đến 100%
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},  # Nguy hiểm
                        {'range': [30, 60], 'color': "orange"},  # Cảnh báo
                        {'range': [60, 100], 'color': "green"}  # An toàn
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': decrease_ratio * 100  # Giá trị hiện tại
                    }
                }
            ))
            col5.plotly_chart(fig_decrease, use_container_width=True)

            # Đánh giá mức độ than tự cháy dựa trên xu hướng và ngưỡng
            max_forecast = np.max(forecast_avg_real)

            st.subheader("🔥 Đánh giá xu hướng nhiệt độ và mức độ than tự cháy")

            # Trường hợp 1: Nhiệt độ có xu hướng tăng rõ rệt (> 60%)
            if increase_ratio > 0.8:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng tăng rõ rệt và vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy nghiêm trọng!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng tăng rõ rệt và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần theo dõi chặt chẽ.")
                else:
                    st.info(
                        "⚠️ Nhiệt độ có xu hướng tăng rõ rệt nhưng vẫn dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            elif 0.8 > increase_ratio > 0.6:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng tăng nhẹ và vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy, cần có biện pháp ngăn chặn!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng tăng nhẹ và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần tiếp tục theo dõi.")
                else:
                    st.info(
                        "⚠️ Nhiệt độ có xu hướng tăng nhẹ nhưng vẫn dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            elif 0.6 >= increase_ratio > 0.5:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng tăng ít và vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy, cần có biện pháp ngăn chặn!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng tăng ít và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần tiếp tục theo dõi.")
                else:
                    st.info(
                        "⚠️ Nhiệt độ có xu hướng tăng ít nhưng vẫn dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            # Trường hợp 3: Nhiệt độ không có xu hướng rõ ràng
            elif 0.5 >= increase_ratio > 0.3:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng giảm nhẹ nhưng vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy nghiêm trọng!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng giảm nhẹ và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần theo dõi chặt chẽ.")
                else:
                    st.info("ℹ️ Nhiệt độ có xu hướng giảm nhẹ và dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            # Trường hợp 3: Nhiệt độ không có xu hướng rõ ràng
            else:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng giảm nhưng vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy nghiêm trọng!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng giảm và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần theo dõi chặt chẽ.")
                else:
                    st.info("ℹ️ Nhiệt độ có xu hướng giảm và dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            # Tiêu đề chính
            st.subheader("📊 Phương pháp đánh giá xu hướng nhiệt độ và mức độ nguy cơ than tự cháy")
            # 1. Mục tiêu
            st.subheader("🎯 Mục tiêu")
            st.markdown("""
            - **Đánh giá xu hướng nhiệt độ** dựa trên kết quả dự báo.
            - **Nhận định mức độ nguy cơ than tự cháy**, giúp đưa ra quyết định kịp thời trong khai thác than.
            """)
            # Thêm biểu tượng và màu sắc để tăng tính trực quan
            st.info("💡 Mục tiêu chính: Phát hiện sớm nguy cơ than tự cháy thông qua xu hướng nhiệt độ.")
            # 2. Phương pháp luận
            st.subheader("📚 Phương pháp luận")
            with st.expander("🔍 Chi tiết phương pháp"):
                st.subheader("Bước 1: Tính toán sự thay đổi nhiệt độ giữa các giờ liên tiếp")
                st.markdown("""
                - Sử dụng **sai phân** (`np.diff`) để tính sự thay đổi nhiệt độ giữa các giờ liên tiếp.
                - Kết quả là một mảng mới chứa sự chênh lệch giữa các giá trị liên tiếp.
                """)
                st.subheader("Bước 2: Đánh giá xu hướng dựa trên phần trăm thay đổi")
                st.markdown("""
                - Nếu **> 70%** các giá trị có xu hướng tăng/giảm, kết luận rằng nhiệt độ đang có xu hướng tương ứng.
                - Phương pháp này phù hợp với dữ liệu thực tế, nơi mà nhiệt độ thường có biến động nhỏ xen kẽ.
                """)
                st.subheader("Bước 3: Phân loại và thông báo dựa trên xu hướng")
                st.markdown("""
                - **Tăng rõ rệt (> 70%):** Cảnh báo nguy cơ than tự cháy gia tăng 🔥.
                - **Giảm rõ rệt (> 70%):** Thông báo tình trạng an toàn ✅.
                - **Không rõ ràng:** Yêu cầu tiếp tục theo dõi ℹ️.
                """)
                st.subheader("Bước 4: Kết hợp với ngưỡng nhiệt độ")
                st.markdown("""
                - **Ngưỡng thấp:** 40°C 🟢.
                - **Ngưỡng cao:** 50°C 🔴.
                - Kết hợp xu hướng nhiệt độ với các ngưỡng để nhận định tình trạng than tự cháy.
                """)
            # 3. Cơ sở khoa học
            st.subheader("🔬 Cơ sở khoa học")
            with st.expander("📖 Chi tiết cơ sở khoa học"):
                st.markdown("""
                - **Nhiệt độ tăng:** Khi nhiệt độ tăng liên tục, phản ánh quá trình oxy hóa than diễn ra mạnh mẽ, dẫn đến nguy cơ tự cháy 🔥.
                - **Nhiệt độ giảm:** Khi nhiệt độ giảm liên tục, cho thấy quá trình oxy hóa bị kiểm soát hoặc môi trường không thuận lợi cho sự tự cháy ❄️.
                - **Nhiệt độ không ổn định:** Sự dao động nhiệt độ có thể do các yếu tố ngoại cảnh (như thay đổi môi trường, hoạt động khai thác) hoặc do dữ liệu chưa đủ chính xác để đưa ra kết luận ⚠️.
                - **Phương pháp đánh giá xu hướng:** Dựa trên sự thay đổi liên tiếp của nhiệt độ là một cách tiếp cận đơn giản nhưng hiệu quả.
                - **Xu hướng tăng/giảm liên tục:** Cung cấp thông tin về tính ổn định của hệ thống, giúp đưa ra quyết định kịp thời ⏳.
                """)
            # 4. Ưu điểm của phương pháp
            st.subheader("✅ Ưu điểm của phương pháp")
            with st.expander("🌟 Chi tiết ưu điểm"):
                st.markdown("""
                - **Đơn giản và dễ hiểu:** Chỉ sử dụng phép tính đơn giản như sai phân và điều kiện logic theo ngưỡng để xác định xu hướng.
                - **Hiệu quả trong thực tế:** Phương pháp này phù hợp với các hệ thống giám sát nhiệt độ trong khai thác than, nơi mà xu hướng nhiệt độ là một yếu tố quan trọng để đánh giá nguy cơ.
                - **Dễ tích hợp với giao diện người dùng:** Các thông báo và biểu đồ trực quan giúp người dùng nhanh chóng nắm bắt tình trạng.
                """)
    else:
        st.warning("Vui lòng nhấn nút Tải dữ liệu và xử lý dữ liệu trước khi huấn luyện hoặc dự báo.")
# Dự báo giá trị lớn nhất
if ForcastMax:
    if 'hourly_max' in st.session_state:
        st.empty()
        model_max, scaler_max = load_model_with_param(f"max_{selected_parameter}")
        if model_max is None or scaler_max is None:
            st.error(f"Không tìm thấy mô hình hoặc scaler cho giá trị lớn nhất của {selected_parameter}!")
            st.stop()
        with st.spinner(f"Đang dự báo giá trị lớn nhất của {selected_parameter}..."):
            # Lấy dữ liệu cuối cùng (49 giờ gần nhất)
            last_49_max = st.session_state.hourly_max[selected_parameter].values[-49:]
            print(last_49_max)
            # Kiểm tra và xử lý giá trị ngoại lệ
            last_49_max = handle_outliers(last_49_max)

            original_value = last_49_max[0]
            last_48_max_diff = make_stationary(last_49_max)
            last_48_max_scaled, _ = normalize_data(last_48_max_diff, scaler_type)
            if last_48_max_scaled.shape[0] < 48:
                st.error(f"Không đủ dữ liệu để dự báo! Kích thước dữ liệu: {last_48_max_scaled.shape[0]} mẫu.")
                st.stop()
            last_48_max_scaled = last_48_max_scaled.reshape((1, 48, 1))
            forecast_max_scaled = model_max.predict(last_48_max_scaled)
            forecast_max_diff = scaler_max.inverse_transform(forecast_max_scaled.reshape(-1, 1))
            forecast_max_real = np.cumsum(forecast_max_diff.flatten()) + original_value
            last_hour = st.session_state.hourly_max["hour"].iloc[-1]
            forecast_time = pd.date_range(start=last_hour, periods=9, freq="h")[1:]

            st.title(f"📊 Dự báo giá trị lớn nhất của {selected_parameter} trong 8 giờ tiếp theo")

            forecast_df = pd.DataFrame({
                "Thời gian": forecast_time,
                "Giá trị dự báo": forecast_max_real
            })

            # Tổ chức giao diện thành các tab
            tab1, tab2 = st.tabs(["Biểu đồ", "Bảng dữ liệu"])

            with tab1:
                # Vẽ biểu đồ với Plotly
                fig = go.Figure()

                # Giá trị thực tế
                actual_values = st.session_state.hourly_max[f"{selected_parameter}"].values[-48:]
                actual_time = st.session_state.hourly_max["hour"].iloc[-48:].values

                # Chuyển đổi actual_time thành datetime
                actual_time = pd.to_datetime(actual_time)

                # Đảm bảo forecast_time bắt đầu ngay sau actual_time[-1]
                if forecast_time[0] != actual_time[-1] + pd.Timedelta(hours=1):
                    forecast_time = pd.date_range(start=actual_time[-1] + pd.Timedelta(hours=1), periods=8, freq="h")

                fig.add_trace(go.Scatter(
                    x=actual_time,
                    y=actual_values,
                    mode='lines+markers',
                    name="Giá trị thực tế",
                    line=dict(color="green", dash="dot"),
                    marker=dict(symbol="x")
                ))

                # Giá trị dự báo
                fig.add_trace(go.Scatter(
                    x=forecast_time,
                    y=forecast_max_real,
                    mode='lines+markers',
                    name="Dự báo",
                    line=dict(color="red", dash="dot"),
                    marker=dict(symbol="circle")
                ))

                # Nét nối giữa giá trị thực tế và dự báo
                connection_x = [actual_time[-1], forecast_time[0]]
                connection_y = [actual_values[-1], forecast_max_real[0]]
                fig.add_trace(go.Scatter(
                    x=connection_x,
                    y=connection_y,
                    mode='lines',
                    name="Nối giá trị",
                    line=dict(color="orange", dash="dot")
                ))

                # Cấu hình layout
                fig.update_layout(
                    title="📉 Giá trị lớn nhất thực tế 48 giờ qua và giá trị dự báo trong 8 giờ tiếp theo",
                    xaxis_title="Thời gian",
                    yaxis_title="Giá trị lớn nhất",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )

                st.plotly_chart(fig)

            with tab2:
                # Hiển thị bảng dữ liệu
                st.write(forecast_df)

            # Hiển thị thông tin thống kê
            #st.subheader("📊 Thông tin thống kê dữ liệu dự báo")
            col1, col2, col3, col4, col5 = st.columns(5)

            # 1. Trung bình
            mean_value = np.mean(forecast_max_real)
            #col1.metric("Trung bình", f"{mean_value:.2f} °C")

            # Thêm biểu đồ gauge cho trung bình
            fig_mean = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mean_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Trung bình"},
                gauge={
                    'axis': {'range': [None, 60]},  # Phạm vi nhiệt độ tối đa
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 50], 'color': "orange"},
                        {'range': [50, 60], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': mean_value
                    }
                }
            ))
            col1.plotly_chart(fig_mean, use_container_width=True)

            # 2. Lớn nhất
            max_value = np.max(forecast_max_real)
            #col2.metric("Lớn nhất", f"{max_value:.2f} °C")

            # Thêm biểu đồ gauge cho giá trị lớn nhất
            fig_max = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Lớn nhất"},
                gauge={
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 50], 'color': "orange"},
                        {'range': [50, 60], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': max_value
                    }
                }
            ))
            col2.plotly_chart(fig_max, use_container_width=True)

            # 3. Nhỏ nhất
            min_value = np.min(forecast_max_real)
            #col3.metric("Nhỏ nhất", f"{min_value:.2f} °C")

            # Thêm biểu đồ gauge cho giá trị nhỏ nhất
            fig_min = go.Figure(go.Indicator(
                mode="gauge+number",
                value=min_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Nhỏ nhất"},
                gauge={
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 50], 'color': "orange"},
                        {'range': [50, 60], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': min_value
                    }
                }
            ))
            col3.plotly_chart(fig_min, use_container_width=True)

            # Đánh giá xu hướng nhiệt độ
            temperature_changes = np.diff(forecast_max_real)  # Sự thay đổi giữa các giờ liên tiếp

            # Tính tỷ lệ tăng/giảm
            positive_changes = np.sum(temperature_changes > 0)  # Số lần tăng
            negative_changes = np.sum(temperature_changes < 0)  # Số lần giảm
            total_changes = len(temperature_changes)

            # Ngưỡng tỷ lệ tăng/giảm (ví dụ: > 60%)
            increase_ratio = positive_changes / total_changes if total_changes > 0 else 0
            decrease_ratio = negative_changes / total_changes if total_changes > 0 else 0

            # Hiển thị tỷ lệ tăng/giảm
            #st.write(f"📊 Tỷ lệ tăng: {increase_ratio * 100:.1f}% | Tỷ lệ giảm: {decrease_ratio * 100:.1f}%")

            # Biểu đồ gauge cho tỷ lệ tăng
            fig_increase = go.Figure(go.Indicator(
                mode="gauge+number",
                value=increase_ratio * 100,  # Chuyển tỷ lệ về phần trăm
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Tỷ lệ tăng (%)"},
                gauge={
                    'axis': {'range': [None, 100]},  # Phạm vi từ 0% đến 100%
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},  # An toàn
                        {'range': [30, 60], 'color': "orange"},  # Cảnh báo
                        {'range': [60, 100], 'color': "red"}  # Nguy hiểm
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': increase_ratio * 100  # Giá trị hiện tại
                    }
                }
            ))
            col4.plotly_chart(fig_increase, use_container_width=True)

            # Biểu đồ gauge cho tỷ lệ giảm
            fig_decrease = go.Figure(go.Indicator(
                mode="gauge+number",
                value=decrease_ratio * 100,  # Chuyển tỷ lệ về phần trăm
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Tỷ lệ giảm (%)"},
                gauge={
                    'axis': {'range': [None, 100]},  # Phạm vi từ 0% đến 100%
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},  # Nguy hiểm
                        {'range': [30, 60], 'color': "orange"},  # Cảnh báo
                        {'range': [60, 100], 'color': "green"}  # An toàn
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': decrease_ratio * 100  # Giá trị hiện tại
                    }
                }
            ))
            col5.plotly_chart(fig_decrease, use_container_width=True)

            # Đánh giá mức độ than tự cháy dựa trên xu hướng và ngưỡng
            max_forecast = np.max(forecast_max_real)

            st.subheader("🔥 Đánh giá xu hướng nhiệt độ và mức độ than tự cháy")

            # Trường hợp 1: Nhiệt độ có xu hướng tăng rõ rệt (> 60%)
            if increase_ratio >= 0.8:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng tăng rõ rệt và vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy nghiêm trọng!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng tăng rõ rệt và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần theo dõi chặt chẽ.")
                else:
                    st.info(
                        "⚠️ Nhiệt độ có xu hướng tăng rõ rệt nhưng vẫn dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            elif 0.8 > increase_ratio > 0.6:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng tăng nhẹ và vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy, cần có biện pháp ngăn chặn!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng tăng nhẹ và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần tiếp tục theo dõi.")
                else:
                    st.info(
                        "⚠️ Nhiệt độ có xu hướng tăng nhẹ nhưng vẫn dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            elif 0.6 >= increase_ratio > 0.5:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng tăng ít và vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy, cần có biện pháp ngăn chặn!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng tăng ít và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần tiếp tục theo dõi.")
                else:
                    st.info(
                        "⚠️ Nhiệt độ có xu hướng tăng ít nhưng vẫn dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            elif 0.5 >= increase_ratio > 0.3:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng giảm nhẹ nhưng vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy nghiêm trọng!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng giảm nhẹ và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần theo dõi chặt chẽ.")
                else:
                    st.info("ℹ️ Nhiệt độ có xu hướng giảm nhẹ và dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")
            # Trường hợp 3: Nhiệt độ không có xu hướng rõ ràng
            else:
                if max_forecast >= 50:
                    st.error(
                        "⚠️ Nhiệt độ có xu hướng giảm nhưng vượt ngưỡng cao (>= 50°C). Nguy cơ than tự cháy nghiêm trọng!")
                elif 40 <= max_forecast < 50:
                    st.warning(
                        "⚠️ Nhiệt độ có xu hướng giảm và nằm trong khoảng ngưỡng cảnh báo (40°C - 50°C). Cần theo dõi chặt chẽ.")
                else:
                    st.info("ℹ️ Nhiệt độ có xu hướng giảm và dưới ngưỡng thấp (< 40°C). Cần tiếp tục theo dõi.")

            # Tiêu đề chính
            st.subheader("📊 Phương pháp đánh giá xu hướng nhiệt độ và mức độ nguy cơ than tự cháy")
            # 1. Mục tiêu
            st.subheader("🎯 Mục tiêu")
            st.markdown("""
                - **Đánh giá xu hướng nhiệt độ** dựa trên kết quả dự báo.
                - **Nhận định mức độ nguy cơ than tự cháy**, giúp đưa ra quyết định kịp thời trong khai thác than.
                """)
            # Thêm biểu tượng và màu sắc để tăng tính trực quan
            st.info("💡 Mục tiêu chính: Phát hiện sớm nguy cơ than tự cháy thông qua xu hướng nhiệt độ.")
            # 2. Phương pháp luận
            st.subheader("📚 Phương pháp luận")
            with st.expander("🔍 Chi tiết phương pháp"):
                st.subheader("Bước 1: Tính toán sự thay đổi nhiệt độ giữa các giờ liên tiếp")
                st.markdown("""
                    - Sử dụng **sai phân** để tính sự thay đổi nhiệt độ giữa các giờ liên tiếp.
                    - Kết quả là một mảng mới chứa sự chênh lệch giữa các giá trị liên tiếp.
                    """)
                st.subheader("Bước 2: Đánh giá xu hướng dựa trên phần trăm thay đổi")
                st.markdown("""
                    - Dựa trên phần trăm thay đổi các giá trị có xu hướng tăng/giảm, kết luận rằng nhiệt độ đang có xu hướng tương ứng.
                    - Phương pháp này phù hợp với dữ liệu thực tế, nơi mà nhiệt độ thường có biến động nhỏ xen kẽ.
                    """)
                st.subheader("Bước 3: Kết hợp với ngưỡng nhiệt độ")
                st.markdown("""
                    - **Ngưỡng thấp:** 40°C 🟢.
                    - **Ngưỡng cao:** 50°C 🔴.
                    - Kết hợp xu hướng nhiệt độ với các ngưỡng để nhận định tình trạng than tự cháy.
                    """)
            # 3. Cơ sở khoa học
            st.subheader("🔬 Cơ sở khoa học")
            with st.expander("📖 Chi tiết cơ sở khoa học"):
                st.markdown("""
                    - **Nhiệt độ tăng:** Khi nhiệt độ tăng liên tục, phản ánh quá trình oxy hóa than diễn ra mạnh mẽ, dẫn đến nguy cơ tự cháy 🔥.
                    - **Nhiệt độ giảm:** Khi nhiệt độ giảm liên tục, cho thấy quá trình oxy hóa bị kiểm soát hoặc môi trường không thuận lợi cho sự tự cháy ❄️.
                    - **Nhiệt độ không ổn định:** Sự dao động nhiệt độ có thể do các yếu tố ngoại cảnh (như thay đổi môi trường, hoạt động khai thác) hoặc do dữ liệu chưa đủ chính xác để đưa ra kết luận ⚠️.
                    - **Phương pháp đánh giá xu hướng:** Dựa trên sự thay đổi liên tiếp của nhiệt độ là một cách tiếp cận đơn giản nhưng hiệu quả.
                    - **Xu hướng tăng/giảm liên tục:** Cung cấp thông tin về tính ổn định của hệ thống, giúp đưa ra quyết định kịp thời ⏳.
                    """)
            # 4. Ưu điểm của phương pháp
            st.subheader("✅ Ưu điểm của phương pháp")
            with st.expander("🌟 Chi tiết ưu điểm"):
                st.markdown("""
                    - **Đơn giản và dễ hiểu:** Chỉ sử dụng phép tính đơn giản như sai phân và điều kiện logic theo ngưỡng để xác định xu hướng.
                    - **Hiệu quả trong thực tế:** Phương pháp này phù hợp với các hệ thống giám sát nhiệt độ trong khai thác than, nơi mà xu hướng nhiệt độ là một yếu tố quan trọng để đánh giá nguy cơ.
                    - **Dễ tích hợp với giao diện người dùng:** Các thông báo và biểu đồ trực quan giúp người dùng nhanh chóng nắm bắt tình trạng.
                    """)
    else:
        st.warning("Vui lòng nhấn nút Tải dữ liệu và xử lý dữ liệu trước khi huấn luyện hoặc dự báo.")
# Dự báo giá trị nhỏ nhất
if ForcastMin:
    if 'hourly_min' in st.session_state:
        st.empty()
        model_min, scaler_min = load_model_with_param(f"min_{selected_parameter}")
        if model_min is None or scaler_min is None:
            st.error(f"Không tìm thấy mô hình hoặc scaler cho giá trị nhỏ nhất của {selected_parameter}!")
            st.stop()
        with st.spinner(f"Đang dự báo giá trị nhỏ nhất của {selected_parameter}..."):
            # Lấy dữ liệu cuối cùng (49 giờ gần nhất)
            last_49_min = st.session_state.hourly_min[selected_parameter].values[-49:]

            # Kiểm tra và xử lý giá trị ngoại lệ
            last_49_min = handle_outliers(last_49_min)

            original_value = last_49_min[0]
            last_48_min_diff = make_stationary(last_49_min)
            last_48_min_scaled, _ = normalize_data(last_48_min_diff,scaler_type)
            if last_48_min_scaled.shape[0] < 48:
                st.error(f"Không đủ dữ liệu để dự báo! Kích thước dữ liệu: {last_48_min_scaled.shape[0]} mẫu.")
                st.stop()
            last_48_min_scaled = last_48_min_scaled.reshape((1, 48, 1))
            forecast_min_scaled = model_min.predict(last_48_min_scaled)
            forecast_min_diff = scaler_min.inverse_transform(forecast_min_scaled.reshape(-1, 1))
            forecast_min_real = np.cumsum(forecast_min_diff.flatten()) + original_value
            last_hour = st.session_state.hourly_min["hour"].iloc[-1]
            forecast_time = pd.date_range(start=last_hour, periods=9, freq="h")[1:]
            st.subheader(f"Dự báo giá trị nhỏ nhất của {selected_parameter} trong 8 giờ tiếp theo")
            forecast_df = pd.DataFrame({
                "Thời gian": forecast_time,
                "Giá trị dự báo": forecast_min_real
            })
            # Tổ chức giao diện thành các tab
            tab1, tab2 = st.tabs(["Biểu đồ", "Bảng dữ liệu"])

            with tab1:
                # Vẽ biểu đồ với Plotly
                fig = go.Figure()

                # Giá trị thực tế
                actual_values = st.session_state.hourly_min[f"{selected_parameter}"].values[-48:]
                actual_time = st.session_state.hourly_min["hour"].iloc[-48:].values

                fig.add_trace(go.Scatter(
                    x=actual_time,
                    y=actual_values,
                    mode='lines+markers',
                    name="Giá trị thực tế",
                    line=dict(color="green", dash="dot"),
                    marker=dict(symbol="x")
                ))

                # Giá trị dự báo
                fig.add_trace(go.Scatter(
                    x=forecast_time,
                    y=forecast_min_real,
                    mode='lines+markers',
                    name="Dự báo",
                    line=dict(color="red"),
                    marker=dict(symbol="circle")
                ))

                # Nét nối giữa giá trị thực tế và dự báo
                connection_x = [actual_time[-1], forecast_time[0]]
                connection_y = [actual_values[-1], forecast_min_real[0]]
                fig.add_trace(go.Scatter(
                    x=connection_x,
                    y=connection_y,
                    mode='lines',
                    name="Nối giá trị",
                    line=dict(color="blue", dash="dot")
                ))

                # Cấu hình layout
                fig.update_layout(
                    title="📉 Giá trị nhỏ nhất thực tế 48 giờ qua và giá trị dự báo trong 8 giờ tiếp theo",
                    xaxis_title="Thời gian",
                    yaxis_title="Giá trị nhỏ nhất",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )

                st.plotly_chart(fig)

            with tab2:
                # Hiển thị bảng dữ liệu
                st.write(forecast_df)

            # Hiển thị thông tin thống kê
            st.subheader("Thông tin thống kê")
            col1, col2, col3 = st.columns(3)
            col1.metric("Trung bình", f"{np.mean(forecast_min_real):.2f}")
            col2.metric("Lớn nhất", f"{np.max(forecast_min_real):.2f}")
            col3.metric("Nhỏ nhất", f"{np.min(forecast_min_real):.2f}")
            # st.write(forecast_df)
            #
            # # Vẽ đồ thị
            # plt.figure(figsize=(10, 6))
            #
            # # Vẽ giá trị thực tế
            # actual_values = st.session_state.hourly_min[f"{selected_parameter}"].values[-48:]
            # actual_time = st.session_state.hourly_min["hour"].iloc[-48:].values  # Chuyển đổi thành mảng NumPy
            #
            # # Kiểm tra kích thước dữ liệu
            # if len(actual_time) == 0 or len(forecast_time) == 0:
            #     st.error("Không đủ dữ liệu để vẽ đồ thị!")
            #     st.stop()
            #
            # plt.plot(actual_time, actual_values, label="Giá trị thực tế", color="green", linestyle="--", marker="x")
            #
            # # Vẽ giá trị dự báo
            # plt.plot(forecast_time, forecast_min_real, label="Dự báo", color="red", marker="o")
            #
            # # Nét nối giữa giá trị thực tế và dự báo
            # connection_x = [actual_time[-1], forecast_time[0]]
            # connection_y = [actual_values[-1], forecast_min_real[0]]
            # plt.plot(connection_x, connection_y, color="blue", linestyle=":", linewidth=2, label="Nối giá trị")
            #
            # # Đặt nhãn trục x
            # plt.xlabel("Thời gian")
            #
            # # Đặt nhãn trục y
            # plt.ylabel("Giá trị nhỏ nhất")
            #
            # # Đặt tiêu đề đồ thị
            # plt.title("So sánh giá trị thực tế và giá trị dự báo trong 8 giờ tiếp theo")
            #
            # # Xoay nhãn trục x để dễ đọc
            # plt.xticks(rotation=45)
            #
            # # Thêm chú thích
            # plt.legend()
            #
            # # Điều chỉnh layout để tránh chồng chéo
            # plt.tight_layout()
            #
            # # Hiển thị đồ thị
            # st.pyplot(plt)
    else:
        st.warning("Vui lòng nhấn nút Tải dữ liệu và xử lý dữ liệu trước khi huấn luyện hoặc dự báo.")