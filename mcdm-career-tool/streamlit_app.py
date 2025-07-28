import streamlit as st
import numpy as np
from pymcdm.methods import TOPSIS
from pymcdm.helpers import rrankdata
from pymcdm.weights import AHP
import matplotlib.pyplot as plt


def build_ahp_weights(pairwise_matrix):
    model = AHP()
    return model(pairwise_matrix)


def apply_time_discounting(weights, discount_rate):
    decay = np.exp(-discount_rate * np.arange(len(weights)))
    discounted = weights * decay
    return discounted / np.sum(discounted)


def rank_with_topsis(matrix, weights, types):
    model = TOPSIS()
    scores = model(matrix, weights, types)
    return rrankdata(scores), scores


# --- Streamlit UI ---
st.set_page_config(page_title="Career Trade-off Analyzer", layout="centered")
st.title("🔧 Career Decision Simulator – AHP + TOPSIS + Discounting")

st.markdown("This tool helps you **rank career paths** based on your preferences, using Multi-Criteria Decision Making (AHP + TOPSIS) with time-discounting.")

# Criteria and types
criteria = ["Income", "Working Hours", "Stability"]
types = np.array([1, -1, 1])  # Profit, Cost, Profit

# Alternatives (modifiable if needed)
labels = [
    "High-Risk High-Reward",
    "Balanced Private Sector",
    "Stable Public Sector",
    "Flexible Low-Income",
    "Mission-Driven Role"
]
decision_matrix = np.array([
    [9, 60, 4],
    [7, 45, 6],
    [5, 40, 9],
    [3, 30, 5],
    [4, 55, 7]
])

st.header("1️⃣ Set Pairwise Importance (AHP)")

col1, col2, col3 = st.columns(3)
c1 = col1.slider("💰 Income vs Hours", 1/9.0, 9.0, 3.0, step=0.5)
c2 = col2.slider("💰 Income vs Stability", 1/9.0, 9.0, 0.5, step=0.5)
c3 = col3.slider("🕐 Hours vs Stability", 1/9.0, 9.0, 1/7.0, step=0.5)

# Build pairwise matrix
ahp_matrix = np.array([
    [1,   c1,  c2],
    [1/c1, 1,  c3],
    [1/c2, 1/c3, 1]
])

weights = build_ahp_weights(ahp_matrix)

st.write("### 🧮 Raw AHP Weights")
st.write({k: f"{v:.3f}" for k, v in zip(criteria, weights)})

# Time discounting
st.header("2️⃣ Choose Time Preference")

discount_rate = st.slider("⏳ Discount Rate (0 = long-term focus, 1 = short-term)", 0.0, 1.0, 0.2, step=0.05)
discounted_weights = apply_time_discounting(weights, discount_rate)

st.write("### 📉 Discounted Weights")
st.write({k: f"{v:.3f}" for k, v in zip(criteria, discounted_weights)})

# Run TOPSIS
ranks, scores = rank_with_topsis(decision_matrix, discounted_weights, types)

st.header("3️⃣ Career Rankings")

result_data = [{"Option": labels[i], "Score": scores[i], "Rank": int(ranks[i])} for i in range(len(labels))]
st.dataframe(result_data, use_container_width=True)

# Plotting
fig, ax = plt.subplots()
ax.barh(labels, scores, color="skyblue")
ax.set_xlabel("TOPSIS Score")
ax.set_title("Career Preference Scores")
st.pyplot(fig)
