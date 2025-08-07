import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mcdm_utils import (
    build_ahp_weights, 
    consistency_ratio,
    apply_time_discounting, 
    rank_with_topsis, 
    load_demo_dataset,
    load_realistic_jobs_dataset,
    normalize_data,
    analyze_results,
    radar_chart
)

# --- Streamlit UI ---
st.set_page_config(page_title="Career Trade-off Analyzer", layout="centered")
st.title("🔧 Career Decision Simulator – AHP + TOPSIS + Discounting")

st.markdown("This tool helps you **rank career paths** based on your preferences, using Multi-Criteria Decision Making (AHP + TOPSIS) with time-discounting.")

# Criteria and types
criteria = ["Income", "Working Hours", "Stability"]
types = np.array([1, -1, 1])  # Benefit, Cost, Benefit

# Default decision matrix
default_decision_matrix = np.array([
    [9, 60, 4],  # High-Risk High-Reward
    [7, 45, 6],  # Balanced Private
    [5, 40, 9],  # Public Sector
    [3, 30, 5],  # Low-Income Flexible
    [4, 55, 7]   # Mission-Driven
])

default_labels = [
    "High-Risk High-Reward",
    "Balanced Private Sector",
    "Stable Public Sector",
    "Flexible Low-Income",
    "Mission-Driven Role"
]

# Demo dataset loader
st.sidebar.header("📊 Data Options")
dataset_option = st.sidebar.selectbox(
    "Choose Dataset",
    ["Manual Entry", "Realistic Jobs Dataset", "Demo Scenarios"],
    index=0
)

if dataset_option == "Realistic Jobs Dataset":
    jobs_df = load_realistic_jobs_dataset()
    st.sidebar.success("✅ Realistic jobs dataset loaded!")
    st.sidebar.markdown("*Based on typical market data and research*")
    
    # Show job descriptions in sidebar
    with st.sidebar.expander("📋 Job Descriptions"):
        for job, desc in jobs_df['Description'].items():
            st.write(f"**{job}:** {desc}")
    
    decision_matrix = jobs_df[['Income', 'Working Hours', 'Stability']].to_numpy()
    labels = jobs_df.index.tolist()
    
elif dataset_option == "Demo Scenarios":
    demo_df = load_demo_dataset()
    st.sidebar.success("Demo scenarios loaded!")
    decision_matrix = demo_df.to_numpy()
    labels = demo_df.index.tolist()
    
else:  # Manual Entry
    # Editable matrix
    st.header("📝 Edit Career Options")
    
    # Add criteria descriptions
    st.markdown("""
    **Criteria Definitions:**
    - **Income:** Expected compensation level (1-10, higher is better)
    - **Working Hours:** Average weekly hours (actual hours, lower is better for work-life balance)  
    - **Stability:** Job security and long-term prospects (1-10, higher means more secure)
    """)
    
    initial_df = pd.DataFrame(default_decision_matrix, columns=criteria, index=default_labels)
    edited_df = st.data_editor(
        initial_df,
        num_rows="dynamic",
        use_container_width=True,
        key="decision_matrix_editor"
    )
    decision_matrix = edited_df.to_numpy()
    labels = edited_df.index.astype(str).tolist()

st.header("1️⃣ Set Pairwise Importance (AHP)")

col1, col2, col3 = st.columns(3)
c1 = col1.slider("💰 Income vs Hours", 1/9.0, 9.0, 3.0, step=0.5)
c2 = col2.slider("💰 Income vs Stability", 1/9.0, 9.0, 0.5, step=0.5)
c3 = col3.slider("🕐 Hours vs Stability", 1/9.0, 9.0, 1/7.0, step=0.5)

# Build pairwise matrix
ahp_matrix = np.array([
    [1,    c1,   c2],
    [1/c1, 1,    c3],
    [1/c2, 1/c3, 1]
])

# 👉 Compute the Consistency Ratio here
cr = consistency_ratio(ahp_matrix)   # <-- ADD THIS LINE

# AHP Consistency feedback
if cr > 0.2:
    st.error(f"⚠️ Consistency Ratio: {cr:.3f} (Bad - > 0.2)")
    st.error("💡 **Improvement Tip:** Try adjusting your pairwise judgments to be more consistent. Your comparisons may contradict each other.")
elif cr > 0.1:
    st.warning(f"⚠️ Consistency Ratio: {cr:.3f} (Borderline - 0.1-0.2)")  
    st.warning("💡 **Suggestion:** Consider reviewing your comparisons for better consistency.")
else:
    st.success(f"✅ Consistency Ratio: {cr:.3f} (Good - < 0.1)")

weights = build_ahp_weights(ahp_matrix)

st.write("### 🧮 Raw AHP Weights")
st.write({k: f"{v:.3f}" for k, v in zip(criteria, weights)})

# Sensitivity Analysis
st.header("2️⃣ Sensitivity Analysis")
boost = st.slider('⚖️ Boost Income weight (%)', -50, 50, 0)
adj_weights = weights.copy()
adj_weights[0] *= 1 + boost/100
adj_weights /= adj_weights.sum()

if boost != 0:
    st.write("### 📊 Adjusted Weights")
    st.write({k: f"{v:.3f}" for k, v in zip(criteria, adj_weights)})

# Time discounting
st.header("3️⃣ Choose Time Preference")
discount_rate = st.slider("⏳ Discount Rate (0 = long-term focus, 1 = short-term)", 0.0, 1.0, 0.2, step=0.05)
discounted_weights = apply_time_discounting(adj_weights, discount_rate)

st.write("### 📉 Discounted Weights")
st.write({k: f"{v:.3f}" for k, v in zip(criteria, discounted_weights)})

# Radar Chart - Raw vs Discounted Weights
fig_radar, ax_radar = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
radar_chart(ax_radar, weights, criteria, "Raw Weights", "blue", 0.25)
radar_chart(ax_radar, discounted_weights, criteria, "Discounted Weights", "red", 0.25)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
ax_radar.set_title("Weight Comparison: Raw vs Discounted", pad=20)

st.pyplot(fig_radar)

# Run TOPSIS
ranks, scores = rank_with_topsis(decision_matrix, discounted_weights, types)

# Generate analysis
analysis = analyze_results(scores, labels, decision_matrix, criteria, discounted_weights)

st.header("4️⃣ Career Rankings & Analysis")

# Results table
result_data = [{"Option": labels[i], "Score": f"{scores[i]:.4f}", "Rank": int(ranks[i])} for i in range(len(labels))]
st.dataframe(result_data, use_container_width=True)

# Detailed Analysis Section
st.subheader("🔎 Result Analysis")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(f"""
    ### 🏆 **Top Recommendation: {analysis['best_option']}**
    
    This option scored **{analysis['best_score']:.4f}**, meaning it's closest to your 'ideal' career based on your current priorities.
    
    **Key Insights:**
    - You prioritized **{analysis['top_criterion']}** most heavily (weight: {analysis['top_criterion_weight']:.3f})
    - **{analysis['best_option']}** performs well on this criterion with a score of {analysis['best_performance']:.1f}
    - The score range across all options is {analysis['score_range']:.3f}, indicating {'significant' if analysis['score_range'] > 0.3 else 'moderate'} differentiation
    """)

with col_right:
    st.info(f"""
    **Runner-up Analysis:**
    
    **Lowest Ranked:** {analysis['worst_option']}
    (Score: {analysis['worst_score']:.4f})
    
    This likely didn't align well with your top priorities.
    """)

# Additional context based on generational research
st.markdown("""
---
### 💡 **Research Context**
Modern career research shows that **Gen Z and Millennials often prioritize work-life balance and meaning over just salary**. 
Our tool accounts for these multi-dimensional preferences, unlike traditional career advice that focuses primarily on compensation.

*Note: 56% of employee burnout is caused by negative work culture and contributes to up to half of workforce turnover.*
""")

# Enhanced bar chart with highlighted best option
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['lightgrey'] * len(scores)
colors[np.argmax(scores)] = 'steelblue'

# Sort by scores for better readability
sorted_indices = np.argsort(scores)
sorted_labels = [labels[i] for i in sorted_indices]
sorted_scores = [scores[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

bars = ax.barh(sorted_labels, sorted_scores, color=sorted_colors)
ax.set_xlabel("TOPSIS Score")
ax.set_title("Career Preference Scores (Sorted by Ranking)")

# Add score labels on bars
for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{score:.3f}', ha='left', va='center', fontweight='bold')

# Add ranking numbers
for i, bar in enumerate(bars):
    rank = len(bars) - i
    ax.text(0.01, bar.get_y() + bar.get_height()/2, 
            f'#{rank}', ha='left', va='center', fontweight='bold', 
            color='white' if rank == 1 else 'black')

plt.tight_layout()
st.pyplot(fig)

# Download functionality
st.header("📥 Export Results")
df_out = pd.DataFrame(result_data)
csv = df_out.to_csv(index=False)
st.download_button(
    label="📥 Download Results as CSV",
    data=csv,
    file_name='career_analysis_results.csv',
    mime='text/csv'
)

# Sidebar with additional info
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Methods Used:**
- **AHP** (Analytic Hierarchy Process) - Saaty 1987
- **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution) - Hwang & Yoon 1981  
- **Exponential Time Discounting** - Novel behavioral economics integration

**Novel Features:**
- Time discounting of criteria weights (models short vs long-term preferences)
- Realistic job market data integration
- Comprehensive result analysis with explanations

**Consistency Guide:**
- CR < 0.1: Good consistency
- CR 0.1-0.2: Borderline (consider adjusting)
- CR > 0.2: Poor consistency (revise judgments)
""")

st.sidebar.markdown("---") 

with st.sidebar.expander("🔬 Research Background"):
    st.markdown("""
    **Academic Foundation:**
    - Yavuz (2016) validated AHP-TOPSIS for career decisions
    - OECD research shows non-wage factors strongly influence life satisfaction
    - Behavioral economics: Time preferences affect decision-making
    
    **Future Extensions:**
    - Fuzzy logic for uncertainty handling
    - Machine learning integration
    - Group decision making
    - Additional criteria (growth, health impact)
    """)

st.sidebar.markdown("*Career Decision Simulator v2.0*")