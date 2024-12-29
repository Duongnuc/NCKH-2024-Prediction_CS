import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('lightgbm_model.pkl')

# Function for scaling proportions to 1m³
def scale_to_one_cubic_meter(materials, densities):
    total_volume = sum(materials[i] / densities[i] for i in range(len(materials)))
    scaling_factor = 1000 / total_volume
    return [round(material * scaling_factor, 6) for material in materials]

# Function to check mix validity
def is_valid_mix(materials, scaled_materials):
    slag, ash, cement = scaled_materials[1], scaled_materials[2], scaled_materials[0]
    ratio = (slag + ash) / (slag + ash + cement)

    # Ratio check
    if ratio < 0.3 or ratio > 0.6:
        return "Tỷ lệ xỉ + tro bay/(xỉ + tro bay + xi măng) ngoài khoảng (0.3-0.6)."

    # Range check
    ranges = {
        "Xi măng": (129, 486),
        "Xỉ": (0, 350),
        "Tro bay": (0, 358),
        "Nước": (105, 240),
        "Phụ gia": (0, 23),
        "Đá": (708, 1232),
        "Cát": (555, 971),
    }
    for i, (key, value) in enumerate(ranges.items()):
        if not (value[0] <= scaled_materials[i] <= value[1]):
            return f"{key} ngoài khoảng ({value[0]}-{value[1]})."

    return "Hợp lệ"

# Streamlit UI
st.title("Thiết kế cấp phối bê tông và dự đoán cường độ")

# Input Section
st.sidebar.header("Nhập thông số vật liệu")
cement = st.sidebar.number_input("Khối lượng xi măng (kg/m³)", value=350.00, step=0.01, format="%.2f")
slag = st.sidebar.number_input("Khối lượng xỉ (kg/m³)", value=100.00, step=0.01, format="%.2f")
ash = st.sidebar.number_input("Khối lượng tro bay (kg/m³)", value=70.00, step=0.01, format="%.2f")
water = st.sidebar.number_input("Khối lượng nước (kg/m³)", value=150.00, step=0.01, format="%.2f")
superplastic = st.sidebar.number_input("Khối lượng phụ gia (kg/m³)", value=7.00, step=0.01, format="%.2f")
coarseagg = st.sidebar.number_input("Khối lượng đá (kg/m³)", value=900.00, step=0.01, format="%.2f")
fineagg = st.sidebar.number_input("Khối lượng cát (kg/m³)", value=850.00, step=0.01, format="%.2f")

# Densities
densities = [3.1, 2.7, 2.29, 1.0, 1.07, 2.74, 2.67]
density_labels = [
    "Khối lượng riêng xi măng (kg/l)",
    "Khối lượng riêng xỉ (kg/l)",
    "Khối lượng riêng tro bay (kg/l)",
    "Khối lượng riêng nước (kg/l)",
    "Khối lượng riêng phụ gia (kg/l)",
    "Khối lượng riêng đá (kg/l)",
    "Khối lượng riêng cát (kg/l)",
]

st.sidebar.header("Nhập khối lượng riêng của vật liệu")
densities = [st.sidebar.number_input(label, value=d, step=0.01, format="%.2f") for label, d in zip(density_labels, densities)]

materials = [cement, slag, ash, water, superplastic, coarseagg, fineagg]

# Scale materials to 1m³
scaled_materials = scale_to_one_cubic_meter(materials, densities)

# Display scaled proportions
st.subheader("Cấp phối sau khi quy đổi (kg/m³):")
scaled_df = pd.DataFrame(
    {"Vật liệu": ["Xi măng", "Xỉ", "Tro bay", "Nước", "Phụ gia", "Đá", "Cát"],
     "Khối lượng (kg/m³)": scaled_materials}
)
st.write(scaled_df)

# Check mix validity
validity = is_valid_mix(materials, scaled_materials)
st.subheader("Kết quả kiểm tra cấp phối:")
if validity == "Hợp lệ":
    st.success("Cấp phối hợp lệ.")
else:
    st.error(validity)

# Predict strength if mix is valid
if validity == "Hợp lệ":
    ages = [3, 7, 28, 91]
    strengths = []
    for age in ages:
        input_data = scaled_materials + [age]
        prediction = model.predict([input_data])[0]
        strengths.append(prediction)

    # Plot results with strength values
    st.subheader("Đường cong phát triển cường độ:")
    plt.figure(figsize=(8, 5))
    for i, age in enumerate(ages):
        plt.text(age, strengths[i], f"{strengths[i]:.1f}", fontsize=9, ha='center', va='bottom')
    plt.plot(ages, strengths, marker='o', linestyle='-', label="Cường độ dự đoán")
    plt.xlabel("Tuổi bê tông (ngày)")
    plt.ylabel("Cường độ nén (MPa)")
    plt.title("Sự phát triển cường độ bê tông")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Calculate cost and economic efficiency
    st.subheader("Tính toán giá thành và hiệu quả kinh tế:")
    st.sidebar.header("Nhập giá thành từng vật liệu (VNĐ/kg)")
    prices = {
        "Xi măng": st.sidebar.number_input("Giá xi măng (VNĐ/kg)", value=1800, step=1),
        "Xỉ": st.sidebar.number_input("Giá xỉ (VNĐ/kg)", value=800, step=1),
        "Tro bay": st.sidebar.number_input("Giá tro bay (VNĐ/kg)", value=600, step=1),
        "Nước": st.sidebar.number_input("Giá nước (VNĐ/kg)", value=20, step=1),
        "Phụ gia": st.sidebar.number_input("Giá phụ gia (VNĐ/kg)", value=25000, step=1),
        "Đá": st.sidebar.number_input("Giá đá (VNĐ/kg)", value=241, step=1),
        "Cát": st.sidebar.number_input("Giá cát (VNĐ/kg)", value=351, step=1),
    }

    # Calculate cost per m³
    total_cost = sum(scaled_materials[i] * price for i, price in enumerate(prices.values()))
    st.write(f"**Giá thành sản xuất 1m³ bê tông:** {total_cost:,.0f} VNĐ")

    # Calculate economic efficiency
    strength_at_91_days = strengths[-1]
    economic_efficiency = total_cost / strength_at_91_days if strength_at_91_days > 0 else 0
    st.markdown(f"<span style='color:red'>**Hiệu quả kinh tế (VNĐ/MPa):** {economic_efficiency:,.0f} VNĐ/MPa</span>", unsafe_allow_html=True)

    # Calculate CO2 emissions
    st.subheader("Tính toán phát thải CO2:")
    st.sidebar.header("Phát thải CO2 mỗi vật liệu (kgCO2/kg)")
    emissions = {
        "Xi măng": st.sidebar.number_input("Phát thải CO2 xi măng (kgCO2/kg)", value=0.931000, step=0.01, format="%.6f"),
        "Xỉ": st.sidebar.number_input("Phát thải CO2 xỉ (kgCO2/kg)", value=0.026500, step=0.01, format="%.6f"),
        "Tro bay": st.sidebar.number_input("Phát thải CO2 tro bay (kgCO2/kg)", value=0.019600, step=0.01, format="%.6f"),
        "Nước": st.sidebar.number_input("Phát thải CO2 nước (kgCO2/kg)", value=0.001960, step=0.01, format="%.6f"),
        "Phụ gia": st.sidebar.number_input("Phát thải CO2 phụ gia (kgCO2/kg)", value=0.250000, step=0.01, format="%.6f"),
        "Đá": st.sidebar.number_input("Phát thải CO2 đá (kgCO2/kg)", value=0.007500, step=0.01, format="%.6f"),
        "Cát": st.sidebar.number_input("Phát thải CO2 cát (kgCO2/kg)", value=0.002600, step=0.01, format="%.6f"),
    }

    # Phase A1 emissions
    a1_emissions = sum(scaled_materials[i] * emissions[mat] for i, mat in enumerate(emissions.keys()))

    # Phase A2 emissions
    st.sidebar.header("Cự li vận chuyển và phát thải vận chuyển (A2)")
    transport = {
        "Xi măng": (st.sidebar.number_input("Cự li vận chuyển xi măng (km)", value=60, step=1), 5.18e-5),
        "Xỉ": (st.sidebar.number_input("Cự li vận chuyển xỉ (km)", value=70, step=1), 5.18e-5),
        "Tro bay": (st.sidebar.number_input("Cự li vận chuyển tro bay (km)", value=170, step=1), 5.18e-5),
        "Nước": (st.sidebar.number_input("Cự li vận chuyển nước (km)", value=0, step=1), 0),
        "Phụ gia": (st.sidebar.number_input("Cự li vận chuyển phụ gia (km)", value=16, step=1), 2.21e-4),
        "Đá": (st.sidebar.number_input("Cự li vận chuyển đá (km)", value=20, step=1), 6.3e-5),
        "Cát": (st.sidebar.number_input("Cự li vận chuyển cát (km)", value=35, step=1), 6.3e-5),
    }
    a2_emissions = sum(
        scaled_materials[i] * dist * rate for i, (dist, rate) in enumerate(transport.values())
    )

    # Phase A3 emissions
    st.sidebar.header("Nhập phát thải sản xuất bê tông (A3)")
    a3_emissions = st.sidebar.number_input("Phát thải CO2 sản xuất bê tông (kgCO2/m³)", value=0.507, step=0.0001, format="%.4f")

    # Total emissions
    total_emissions = a1_emissions + a2_emissions + a3_emissions

    # Display emissions breakdown
    st.write(f"**Phát thải CO2 giai đoạn A1:** {a1_emissions:.6f} kgCO2/m³")
    st.write(f"**Phát thải CO2 giai đoạn A2:** {a2_emissions:.6f} kgCO2/m³")
    st.write(f"**Phát thải CO2 giai đoạn A3:** {a3_emissions:.6f} kgCO2/m³")
    st.write(f"**Tổng phát thải CO2:** {total_emissions:.6f} kgCO2/m³")

    # Emissions per MPa
    co2_per_mpa = total_emissions / strength_at_91_days if strength_at_91_days > 0 else 0
    st.markdown(f"<span style='color:red'>**Phát thải CO2 (kgCO2/MPa):** {co2_per_mpa:.6f} kgCO2/MPa</span>", unsafe_allow_html=True)