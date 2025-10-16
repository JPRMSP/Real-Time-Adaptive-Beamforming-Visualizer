import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Adaptive Beamforming Visualizer", layout="centered")

st.title("📡 Real-Time Adaptive Beamforming Visualizer")
st.markdown("""
Simulate how an **Adaptive Antenna Array** dynamically adjusts its beam pattern 
towards a desired direction while suppressing interference — all computed in real-time using the **LMS algorithm**.
""")

# Parameters
num_elements = st.slider("Number of Antenna Elements", 4, 16, 8)
d = 0.5  # spacing in wavelengths
signal_angle = st.slider("Desired Signal Angle (°)", -90, 90, 30)
interference_angle = st.slider("Interference Angle (°)", -90, 90, -40)
snr = st.slider("Signal-to-Noise Ratio (dB)", 0, 30, 20)
mu = st.slider("LMS Step Size", 0.001, 0.1, 0.01)

# Derived parameters
angles = np.linspace(-90, 90, 720)
k = 2 * np.pi
n = np.arange(num_elements)
a_signal = np.exp(1j * k * d * n * np.sin(np.deg2rad(signal_angle)))
a_interf = np.exp(1j * k * d * n * np.sin(np.deg2rad(interference_angle)))

# Simulate received signal + interference + noise
signal = a_signal * np.exp(1j * k * np.random.rand())
interference = a_interf * np.exp(1j * k * np.random.rand())
noise = (np.random.randn(num_elements) + 1j * np.random.randn(num_elements)) * 10**(-snr/20)
x = signal + interference + noise

# LMS Adaptive Beamforming
w = np.zeros(num_elements, dtype=complex)
y_out = []
for i in range(100):
    y = np.dot(np.conj(w), x)
    e = np.exp(1j * k * np.sin(np.deg2rad(signal_angle))) - y
    w = w + mu * x * np.conj(e)
    y_out.append(np.abs(y))

# Compute final beam pattern
pattern = []
for ang in angles:
    steering_vector = np.exp(1j * k * d * n * np.sin(np.deg2rad(ang)))
    response = np.abs(np.dot(np.conj(w), steering_vector))
    pattern.append(response)
pattern = np.array(pattern) / np.max(pattern)

# Visualization
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(np.deg2rad(angles), pattern)
ax.set_title("Adaptive Beam Pattern", va='bottom')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.grid(True)

st.pyplot(fig)

st.markdown("""
### 💡 Observations:
- The **main beam** automatically aligns with the desired signal direction.
- The **interference direction** is suppressed (null steering).
- Adjust **LMS step size** or **antenna count** to observe convergence speed and sharpness.
""")

st.caption("Adaptive Antenna Arrays - FI1981 | Anna University (2021 Regulation)")
