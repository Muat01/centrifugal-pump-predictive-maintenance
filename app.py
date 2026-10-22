import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. إعدادات الصفحة
st.set_page_config(page_title="Smart Pump Dashboard", page_icon="💧", layout="wide")

st.title("💧 Smart Factory: Centrifugal Pump Predictive Maintenance")
st.markdown("لوحة تحكم ذكية لمراقبة أداء مضخة الطرد المركزي، الكشف المبكر عن التسريب، وصيانة المعدات.")

# 2. تحميل البيانات
@st.cache_data
def load_data():
    # استرجاع بيانات الحساسات من مجلد data
    df = pd.read_csv("data/pump_sensor_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# 3. لوحة المؤشرات الرئيسية (KPIs) في أعلى الصفحة
st.subheader("📊 مؤشرات الحساسات الحية (Real-time Sensors Summary)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_pressure = df['pressure_psi'].mean()
    st.metric(label="متوسط الضغط (Pressure)", value=f"{avg_pressure:.1f} psi", delta="-3.2 psi")

with col2:
    avg_flow = df['flow_gpm'].mean()
    st.metric(label="متوسط التدفق (Flow)", value=f"{avg_flow:.1f} gpm", delta="-5.1 gpm")

with col3:
    avg_vibration = df['vibration_mms'].mean()
    st.metric(label="متوسط الاهتزاز (Vibration)", value=f"{avg_vibration:.2f} mm/s", delta="+0.8 mm/s")

with col4:
    current_status = df['status'].iloc[-1]
    status_color = "🔴 تسريب خطر" if current_status == "Leakage" else "🟢 وضع طبيعي"
    st.metric(label="حالة المضخة الحالية", value=status_color)

st.markdown("---")

# 4. الرسوم البيانية للتسمني الزمني (Time-Series Plots)
st.subheader("📈 تتبع قراءات الحساسات واكتشاف الخلل")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
leakage_time = df[df['status'] == 'Leakage']['timestamp'].iloc[0]

# الضغط
ax1.plot(df['timestamp'], df['pressure_psi'], color='blue', label='Pressure (psi)')
ax1.axvline(x=leakage_time, color='red', linestyle='--', label='Leakage Detected')
ax1.set_ylabel('Pressure')
ax1.legend()
ax1.grid(True)

# التدفق
ax2.plot(df['timestamp'], df['flow_gpm'], color='green', label='Flow (gpm)')
ax2.axvline(x=leakage_time, color='red', linestyle='--')
ax2.set_ylabel('Flow')
ax2.legend()
ax2.grid(True)

# الاهتزاز
ax3.plot(df['timestamp'], df['vibration_mms'], color='orange', label='Vibration (mm/s)')
ax3.axvline(x=leakage_time, color='red', linestyle='--')
ax3.set_ylabel('Vibration')
ax3.set_xlabel('Timestamp')
ax3.legend()
ax3.grid(True)

st.pyplot(fig)

# 5. تنبيهات الصيانة التنبؤية
st.subheader("🛠️ توصيات نظام الصيانة التنبؤية (AI Recommendations)")
if current_status == "Leakage" or df['vibration_mms'].iloc[-1] > 3.0:
    st.error("⚠️ **تحذير خطير:** تم رصد انخفاض مفاجئ في الضغط مع ارتفاع في الاهتزاز. يُنصح بإيقاف المضخة فوراً وفحص الأنبوب الخارجي لتجنب تلف الدوار (Impeller).")
else:
    st.success("✅ **النظام مستقر:** جميع القراءات ضمن النطاق الطبيعي. لا توجد حاجة لصيانة فورية.")
