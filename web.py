# ============================================================
# FILE: web.py
# HƯỚNG DẪN CHẠY: 
# Bước 1: nhấp chuột phải vào file web.py -> Open in Integrated Terminal 
# Bước 2: gõ streamlit run web.py vào Terminal
# CÀI: pip install streamlit joblib scikit-learn xgboost pandas
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
 
# ── CẤU HÌNH TRANG ───────────────────────────────────────────
st.set_page_config(
    page_title="Dự đoán Rủi ro Vỡ Nợ",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── CSS CUSTOM ────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── FONT ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── NỀN TỔNG THỂ ── */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8edf5 100%);
    }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2137 0%, #1a3a5c 60%, #1f4e79 100%);
        border-right: 3px solid #2e6da4;
    }
    section[data-testid="stSidebar"] * {
        color: #e8f0f7 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #c8dff0 !important;
        font-weight: 500;
    }

    /* ── HEADER ── */
    .main-header {
        background: linear-gradient(135deg, #0d2137 0%, #1f4e79 50%, #2e6da4 100%);
        color: white !important;
        text-align: center;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(31, 78, 121, 0.3);
    }

    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #1f4e79, #2e6da4) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 3px 10px rgba(31, 78, 121, 0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #163a5a, #1f4e79) !important;
        box-shadow: 0 5px 15px rgba(31, 78, 121, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ── INPUT / SELECTBOX ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider {
        border-color: #2e6da4 !important;
        border-radius: 8px !important;
    }
    .stSlider > div > div > div > div {
        background: #1f4e79 !important;
    }

    /* ── SLIDER ── */
    div[data-testid="stThumbValue"] {
        display: none !important;
    }
    div[data-baseweb="tooltip"] {
        display: none !important;
    }
    .stSlider [data-testid="stTickBar"] {
        display: none !important;
    }

    /* ── METRIC CARDS ── */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #d0e4f5;
        border-left: 4px solid #1f4e79 !important;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(31, 78, 121, 0.1);
    }
    [data-testid="metric-container"] label {
        color: #1f4e79 !important;
        font-weight: 600 !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #0d2137 !important;
        font-weight: 700 !important;
    }

    /* ── KẾT QUẢ DỰ ĐOÁN ── */
    .result-safe {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #155724;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
    }
    .result-risk {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #721c24;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);
    }

    /* ── DATAFRAME ── */
    .stDataFrame {
        border: 1px solid #d0e4f5 !important;
        border-radius: 8px !important;
    }

    /* ── DIVIDER ── */
    hr {
        border-color: #2e6da4 !important;
        opacity: 0.3;
    }

    /* ── SUCCESS / WARNING / ERROR ── */
    .stSuccess {
        background: #d4edda !important;
        border-left: 4px solid #28a745 !important;
        border-radius: 8px !important;
    }
    .stWarning {
        background: #fff3cd !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 8px !important;
    }
    .stError {
        background: #f8d7da !important;
        border-left: 4px solid #dc3545 !important;
        border-radius: 8px !important;
    }

    /* ── EXPANDER / INFO BOX ── */
    .stInfo {
        background: #e8f0f7 !important;
        border-left: 4px solid #1f4e79 !important;
        border-radius: 8px !important;
        color: #0d2137 !important;
    }

    /* ── SUBHEADER ── */
    h2, h3 {
        color: #1f4e79 !important;
        font-weight: 700 !important;
        border-left: 4px solid #2e6da4;
        padding-left: 0.6rem;
    }

    /* ── TABLE ── */
    thead tr th {
        background: #1f4e79 !important;
        color: white !important;
    }
    tbody tr:nth-child(even) {
        background: #f0f4f8 !important;
    }
</style>
""", unsafe_allow_html=True)
 
# ── NẠP MÔ HÌNH ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load('models/best_model.pkl')
    scaler        = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names
 
try:
    model, scaler, feature_names = load_model()
    model_loaded = True
except:
    model_loaded = False
 
# ── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="main-header">🏦 Hệ thống Dự đoán Rủi ro Vỡ Nợ Tín Dụng</div>',
            unsafe_allow_html=True)
 
if not model_loaded:
    st.error("⚠️ Không tìm thấy file mô hình! Hãy chạy Phần 1–5 trong Jupyter trước.")
    st.stop()
 
st.success("✅ Mô hình đã được tải thành công!")
 
# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank.png", width=80)
st.sidebar.title("⚙️ Điều hướng")
tab_choice = st.sidebar.radio(
    "Chọn chức năng:",
    ["🔍 Dự đoán đơn lẻ", "📂 Dự đoán hàng loạt", "📊 Thông tin mô hình"]
)
 
st.sidebar.markdown("---")
st.sidebar.markdown("**📌 Hướng dẫn:**")
st.sidebar.markdown("- **Đơn lẻ**: Nhập thông tin khách hàng → nhận kết quả ngay")
st.sidebar.markdown("- **Hàng loạt**: Upload file CSV → dự đoán toàn bộ")
st.sidebar.markdown("- **Mô hình**: Xem thông tin thuật toán sử dụng")
 
# ── HÀM TIỀN XỬ LÝ ──────────────────────────────────────────
def preprocess_input(data: pd.DataFrame) -> np.ndarray:
    df = data.copy()
    # Encoding
    encode_map = {
        'person_home_ownership':      {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3},
        'loan_intent':                {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2,
                                       'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5},
        'loan_grade':                 {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
        'cb_person_default_on_file':  {'N': 0, 'Y': 1}
    }
    for col, mapping in encode_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
 
    df = df[feature_names]
    return scaler.transform(df)
 
def predict_single(input_df: pd.DataFrame):
    X = preprocess_input(input_df)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return pred, prob
 
# ════════════════════════════════════════════════════════════
# TAB 1: DỰ ĐOÁN ĐƠN LẺ
# ════════════════════════════════════════════════════════════
if tab_choice == "🔍 Dự đoán đơn lẻ":
    st.subheader("🔍 Nhập thông tin khách hàng")
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        st.markdown("**👤 Thông tin cá nhân**")
        col_age1, col_age2 = st.columns([4, 1])
        with col_age1:
            person_age = st.slider("Tuổi", 18, 79, 30, label_visibility="visible")
        with col_age2:
            st.markdown(f"""<div style='background:#1f4e79;color:white;border-radius:8px;
                padding:6px 10px;text-align:center;font-weight:700;font-size:1.1rem;
                margin-top:24px;'>{person_age}</div>""", unsafe_allow_html=True)
        person_income       = st.number_input("Thu nhập hàng năm (USD)",
                                               min_value=5000, max_value=999999,
                                               value=50000, step=1000)
        col_emp1, col_emp2 = st.columns([4, 1])
        with col_emp1:
            person_emp_length = st.slider("Số năm đi làm", 0, 59, 5)
        with col_emp2:
            st.markdown(f"""<div style='background:#1f4e79;color:white;border-radius:8px;
                padding:6px 10px;text-align:center;font-weight:700;font-size:1.1rem;
                margin-top:24px;'>{person_emp_length}</div>""", unsafe_allow_html=True)
        home_options = {
            'RENT – Thuê nhà':          'RENT',
            'OWN – Sở hữu nhà':        'OWN',
            'MORTGAGE – Nhà thế chấp': 'MORTGAGE',
            'OTHER – Khác':             'OTHER',
        }
        home_label = st.selectbox(
            "Tình trạng nhà ở",
            list(home_options.keys()),
            help="Tình trạng sở hữu nhà của người vay"
        )
        person_home_ownership = home_options[home_label]
 
    with col2:
        st.markdown("**💰 Thông tin khoản vay**")
        intent_options = {
            'PERSONAL – Chi tiêu cá nhân':         'PERSONAL',
            'EDUCATION – Học phí / Giáo dục':      'EDUCATION',
            'MEDICAL – Chi phí y tế':               'MEDICAL',
            'VENTURE – Kinh doanh / Khởi nghiệp':  'VENTURE',
            'HOMEIMPROVEMENT – Sửa chữa nhà':       'HOMEIMPROVEMENT',
            'DEBTCONSOLIDATION – Trả nợ gộp':       'DEBTCONSOLIDATION',
        }
        intent_label = st.selectbox(
            "Mục đích vay",
            list(intent_options.keys()),
            help="Mục đích sử dụng khoản vay"
        )
        loan_intent = intent_options[intent_label]
        grade_options = {
            'A – Xuất sắc (rủi ro rất thấp, lãi ~5-7%)':  'A',
            'B – Tốt (rủi ro thấp, lãi ~8-10%)':           'B',
            'C – Trung bình (lãi ~11-13%)':                 'C',
            'D – Dưới trung bình (lãi ~14-16%)':            'D',
            'E – Kém (rủi ro cao, lãi ~17-19%)':            'E',
            'F – Rất kém (rủi ro rất cao, lãi ~20-22%)':   'F',
            'G – Tệ nhất (rủi ro cực cao, lãi ~23-25%)':   'G',
        }
        grade_label = st.selectbox(
            "Hạng tín dụng",
            list(grade_options.keys()),
            help="Mức độ đánh giá rủi ro tín dụng – A tốt nhất, G xấu nhất"
        )
        loan_grade = grade_options[grade_label]
        loan_amnt           = st.number_input("Số tiền vay (USD)",
                                               min_value=500, max_value=35000,
                                               value=10000, step=500)
        col_int1, col_int2 = st.columns([4, 1])
        with col_int1:
            loan_int_rate = st.slider("Lãi suất (%)", 5.0, 25.0, 11.0, step=0.1)
        with col_int2:
            st.markdown(f"""<div style='background:#1f4e79;color:white;border-radius:8px;
                padding:6px 10px;text-align:center;font-weight:700;font-size:1.1rem;
                margin-top:24px;'>{loan_int_rate}%</div>""", unsafe_allow_html=True)
 
    with col3:
        st.markdown("**📋 Thông tin tín dụng**")
        loan_percent_income = round(loan_amnt / person_income, 2)
        st.metric("Tỷ lệ vay/thu nhập", f"{loan_percent_income:.2%}",
                  help="Tự tính từ số tiền vay và thu nhập")
 
        default_options = {
            'N – Chưa từng vỡ nợ': 'N',
            'Y – Đã từng vỡ nợ':   'Y',
        }
        default_label = st.selectbox(
            "Lịch sử vỡ nợ trước đây",
            list(default_options.keys()),
            help="Lịch sử vỡ nợ được ghi nhận tại tổ chức tín dụng"
        )
        cb_person_default_on_file = default_options[default_label]
        col_cred1, col_cred2 = st.columns([4, 1])
        with col_cred1:
            cb_person_cred_hist_length = st.slider("Số năm lịch sử tín dụng", 0, 30, 5)
        with col_cred2:
            st.markdown(f"""<div style='background:#1f4e79;color:white;border-radius:8px;
                padding:6px 10px;text-align:center;font-weight:700;font-size:1.1rem;
                margin-top:24px;'>{cb_person_cred_hist_length}</div>""", unsafe_allow_html=True)
 
    st.markdown("---")
 
    if st.button("🚀 Dự đoán ngay", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'person_age':                   person_age,
            'person_income':                person_income,
            'person_home_ownership':        person_home_ownership,
            'person_emp_length':            person_emp_length,
            'loan_intent':                  loan_intent,
            'loan_grade':                   loan_grade,
            'loan_amnt':                    loan_amnt,
            'loan_int_rate':                loan_int_rate,
            'loan_percent_income':          loan_percent_income,
            'cb_person_default_on_file':    cb_person_default_on_file,
            'cb_person_cred_hist_length':   cb_person_cred_hist_length
        }])
 
        pred, prob = predict_single(input_data)
 
        st.markdown("### 📊 Kết quả dự đoán")
        r1, r2, r3 = st.columns(3)
 
        with r1:
            if pred == 0:
                st.markdown('<div class="result-safe">✅ KHÔNG VỠ NỢ</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-risk">⚠️ CÓ RỦI RO VỠ NỢ</div>',
                            unsafe_allow_html=True)
 
        with r2:
            st.metric("Xác suất Không vỡ nợ", f"{prob[0]*100:.1f}%")
            st.metric("Xác suất Vỡ nợ",       f"{prob[1]*100:.1f}%")
 
        with r3:
            # Gauge chart xác suất rủi ro
            fig, ax = plt.subplots(figsize=(4, 3))
            color = 'tomato' if pred == 1 else 'seagreen'
            ax.barh(['Rủi ro'], [prob[1]*100], color=color,
                    edgecolor='white', height=0.4)
            ax.barh(['Rủi ro'], [100 - prob[1]*100],
                    left=[prob[1]*100], color='#e0e0e0',
                    edgecolor='white', height=0.4)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Xác suất (%)')
            ax.set_title(f'Mức rủi ro: {prob[1]*100:.1f}%', fontsize=11)
            ax.axvline(50, color='gray', linestyle='--', linewidth=1)
            st.pyplot(fig)
            plt.close()
 
        # Khuyến nghị
        st.markdown("### 💡 Khuyến nghị")
        if pred == 0:
            if prob[1] < 0.2:
                st.success("🟢 Rủi ro rất thấp – Có thể phê duyệt khoản vay.")
            else:
                st.warning("🟡 Rủi ro trung bình – Cần xem xét thêm điều kiện vay.")
        else:
            if prob[1] > 0.7:
                st.error("🔴 Rủi ro rất cao – Không nên phê duyệt khoản vay.")
            else:
                st.error("🟠 Rủi ro cao – Cần thêm tài sản thế chấp hoặc bảo lãnh.")
 
# ════════════════════════════════════════════════════════════
# TAB 2: DỰ ĐOÁN HÀNG LOẠT
# ════════════════════════════════════════════════════════════
elif tab_choice == "📂 Dự đoán hàng loạt":
    st.subheader("📂 Upload file CSV để dự đoán hàng loạt")
 
    # Template download
    sample = pd.DataFrame([{
        'person_age': 30, 'person_income': 50000,
        'person_home_ownership': 'RENT', 'person_emp_length': 5,
        'loan_intent': 'PERSONAL', 'loan_grade': 'B',
        'loan_amnt': 10000, 'loan_int_rate': 11.0,
        'loan_percent_income': 0.2,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 5
    }])
 
    st.download_button(
        label="⬇️ Tải file CSV mẫu",
        data=sample.to_csv(index=False).encode('utf-8'),
        file_name='template_credit_risk.csv',
        mime='text/csv'
    )
 
    uploaded_file = st.file_uploader(
        "📤 Upload file CSV của bạn",
        type=['csv'],
        help="File cần có đúng các cột như file mẫu"
    )
 
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**📋 Dữ liệu upload:** {df_upload.shape[0]} dòng × {df_upload.shape[1]} cột")
        st.dataframe(df_upload.head(), use_container_width=True)
 
        if st.button("🚀 Dự đoán toàn bộ", type="primary", use_container_width=True):
            try:
                # Tính loan_percent_income nếu chưa có
                if 'loan_percent_income' not in df_upload.columns:
                    df_upload['loan_percent_income'] = (
                        df_upload['loan_amnt'] / df_upload['person_income']
                    ).round(2)
 
                X_batch = preprocess_input(df_upload)
                preds   = model.predict(X_batch)
                probs   = model.predict_proba(X_batch)
 
                df_result = df_upload.copy()
                df_result['Dự đoán']            = preds
                df_result['Kết quả']            = df_result['Dự đoán'].map(
                    {0: '✅ Không vỡ nợ', 1: '⚠️ Vỡ nợ'})
                df_result['Xác suất vỡ nợ (%)'] = (probs[:, 1] * 100).round(1)
 
                # Thống kê
                st.markdown("### 📊 Kết quả tổng hợp")
                c1, c2, c3 = st.columns(3)
                total    = len(df_result)
                n_safe   = (preds == 0).sum()
                n_risk   = (preds == 1).sum()
 
                c1.metric("Tổng khách hàng", f"{total:,}")
                c2.metric("✅ Không vỡ nợ",  f"{n_safe:,} ({n_safe/total*100:.1f}%)")
                c3.metric("⚠️ Có rủi ro",    f"{n_risk:,} ({n_risk/total*100:.1f}%)")
 
                # Biểu đồ phân phối xác suất
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
                axes[0].hist(probs[:, 1] * 100, bins=30,
                             color='steelblue', edgecolor='white', alpha=0.85)
                axes[0].axvline(50, color='red', linestyle='--', linewidth=1.5,
                                label='Ngưỡng 50%')
                axes[0].set_title('Phân phối xác suất vỡ nợ')
                axes[0].set_xlabel('Xác suất vỡ nợ (%)')
                axes[0].set_ylabel('Số khách hàng')
                axes[0].legend()
 
                axes[1].pie([n_safe, n_risk],
                            labels=['Không vỡ nợ', 'Vỡ nợ'],
                            colors=['seagreen', 'tomato'],
                            autopct='%1.1f%%', startangle=90,
                            explode=(0, 0.05))
                axes[1].set_title('Tỷ lệ kết quả dự đoán')
 
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
 
                # Bảng kết quả
                st.markdown("### 📋 Chi tiết kết quả")
                st.dataframe(
                    df_result[['Kết quả', 'Xác suất vỡ nợ (%)',
                               'loan_grade', 'loan_amnt',
                               'loan_int_rate', 'person_income']],
                    use_container_width=True
                )
 
                # Download kết quả
                st.download_button(
                    label="⬇️ Tải kết quả CSV",
                    data=df_result.to_csv(index=False).encode('utf-8'),
                    file_name='credit_risk_predictions.csv',
                    mime='text/csv'
                )
 
            except Exception as e:
                st.error(f"❌ Lỗi xử lý: {e}")
                st.info("💡 Hãy kiểm tra file CSV có đúng định dạng như file mẫu không.")
 
# ════════════════════════════════════════════════════════════
# TAB 3: THÔNG TIN MÔ HÌNH
# ════════════════════════════════════════════════════════════
elif tab_choice == "📊 Thông tin mô hình":
    st.subheader("📊 Thông tin mô hình đang sử dụng")
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🤖 Thuật toán:**")
        st.info(f"{type(model).__name__}")
 
        st.markdown("**📐 Đặc trưng đầu vào:**")
        for i, feat in enumerate(feature_names, 1):
            st.write(f"  {i}. `{feat}`")
 
    with col2:
        st.markdown("**📈 Hiệu suất mô hình (trên tập test):**")
        st.markdown("""
        | Chỉ số | Giá trị |
        |--------|---------|
        | Accuracy | ~93.6% |
        | F1-Score | ~84.5% |
        | ROC-AUC  | ~97.7% |
        """)
 
        st.markdown("**⚙️ Quy trình xử lý:**")
        st.markdown("""
        1. Lọc outlier (age < 80, emp_length < 60)
        2. Điền missing bằng median
        3. Label Encoding biến phân loại
        4. Chuẩn hóa StandardScaler
        5. Hyperparameter Tuning (RandomizedSearchCV)
        """)
 
    st.markdown("---")
    st.markdown("### 🏫 Thông tin đồ án")
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**👥 Thông tin nhóm:**")
        st.markdown("""
| | |
|---|---|
| **Sinh viên 1** | Nguyễn Huỳnh Nhã Phương |
| **MSSV** | 2311556977 |
| **Sinh viên 2** | Hứa Thị Dĩa |
| **MSSV** | 2311557026 |
| **Giảng viên HD** | ThS. Nguyễn Huỳnh Thông |
| **Trường** | Đại học Nguyễn Tất Thành |
| **Khoa** | Công nghệ Thông tin |
| **Chuyên ngành** | Khoa học Dữ liệu |
| **Lớp** | 23DTH2A |
| **Môn học** | Đồ án Chuyên ngành |
        """)
 
    with col_b:
        st.markdown("**📌 Thông tin đề tài:**")
        st.info("""Tên đề tài: Phân tích và dự đoán rủi ro vỡ nợ của khách hàng vay tín dụng dựa trên hồ sơ tài chính cá nhân sử dụng thuật toán Ensemble Learning
 
Lĩnh vực: Machine Learning – Financial Analytics
Dataset: Credit Risk Dataset (Kaggle) – 32.581 dòng x 12 cột
Bài toán: Phân loại nhị phân (0: Không vỡ nợ | 1: Vỡ nợ)""")
 
    st.markdown("---")
    st.markdown("### 📝 Mô tả đề tài")
    st.markdown("""
Đề tài tập trung vào việc **phân tích dữ liệu hồ sơ tài chính cá nhân** của khách hàng vay tín dụng và **xây dựng mô hình học máy** có khả năng dự đoán rủi ro vỡ nợ, hỗ trợ các tổ chức tài chính đưa ra quyết định cấp tín dụng chính xác hơn.
 
**Mục tiêu chính:**
- Khám phá và phân tích các yếu tố ảnh hưởng đến rủi ro vỡ nợ
- Xây dựng và so sánh hiệu suất các mô hình Ensemble Learning
- Tối ưu hóa mô hình bằng Hyperparameter Tuning
- Triển khai hệ thống dự đoán trực quan trên nền tảng Web
    """)
 
    st.markdown("---")
    st.markdown("### 📊 Kết quả mô hình")
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Trước Hyperparameter Tuning:**")
        st.markdown("""
| Mô hình | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Random Forest | 93.4% | 84.0% | 97.4% |
| XGBoost | 93.6% | 84.5% | 97.7% |
| Gradient Boosting | 93.2% | 83.5% | 97.5% |
        """)
    with col_d:
        st.markdown("**Sau Hyperparameter Tuning:**")
        st.markdown("""
| Mô hình | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Random Forest | 93.7% | 84.8% | 97.6% |
| **XGBoost** | **94.1%** | **85.3%** | **98.0%** |
| Gradient Boosting | 93.5% | 84.2% | 97.7% |
        """)
 
    st.success("🏆 Mô hình tốt nhất: XGBoost sau Hyperparameter Tuning – ROC-AUC: 98.0%")
 
    st.markdown("---")
    st.markdown("### ⚙️ Quy trình thực hiện")
    steps = {
        "1. Thu thập dữ liệu":       "Credit Risk Dataset - Kaggle (32.581 dòng, 12 cột)",
        "2. Khám phá dữ liệu (EDA)": "Phân tích phân phối, tương quan, outlier, mất cân bằng lớp",
        "3. Tiền xử lý":             "Lọc outlier, điền missing (median), Label Encoding, StandardScaler",
        "4. Xây dựng mô hình":       "Random Forest, XGBoost, Gradient Boosting",
        "5. Đánh giá mô hình":       "Accuracy, F1-Score, ROC-AUC, Confusion Matrix",
        "6. Tối ưu hóa":             "RandomizedSearchCV + GridSearchCV (Hyperparameter Tuning)",
        "7. Triển khai":             "Web App Streamlit – dự đoán đơn lẻ và hàng loạt",
    }
    for step, desc in steps.items():
        st.markdown(f"**{step}:** {desc}")
 
    st.markdown("---")
    st.caption("© 2026 – Đồ án Chuyên ngành | Khoa CNTT – Đại học Nguyễn Tất Thành")