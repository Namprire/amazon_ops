Career Decision Analyzer 🔧
A comprehensive multi-criteria decision-making tool that helps you rank career options using AHP (Analytic Hierarchy Process), TOPSIS, and behavioral economics-inspired time discounting.

🌟 Key Features
📊 Multiple Data Sources: Manual entry, realistic jobs dataset, or demo scenarios
🧮 AHP Weight Calculation: Pairwise comparisons with real-time consistency checking
⚖️ Sensitivity Analysis: Dynamic weight adjustments to test robustness
⏳ Time Discounting: Novel behavioral economics integration for short vs long-term preferences
📈 Advanced Analytics: Comprehensive result explanations and insights
🎯 Realistic Job Data: 10+ actual career options with market-based attributes
📊 Enhanced Visualizations: Sorted rankings, radar charts, and highlighted recommendations
📥 Export Functionality: Download detailed analysis results
🚀 Novel Contributions
Time Discounting Integration: First MCDM tool to apply exponential time discounting to criteria weights
Behavioral Economics: Models how people actually make career decisions (short vs long-term focus)
Comprehensive Analysis: Goes beyond numbers to provide meaningful explanations
Real Job Market Data: Grounded in actual career options rather than abstract scenarios
Installation 📦
bash
# Clone the repository
git clone <your-repo-url>
cd career-decision-analyzer

# Install required packages  
pip install streamlit pymcdm matplotlib pandas numpy

# Create data directory and add realistic jobs dataset
mkdir data
# Copy jobs data to data/demo.csv
Usage 🚀
bash
# Run the Streamlit app
streamlit run app.py
File Structure 📁
career-decision-analyzer/
├── app.py                 # Enhanced Streamlit application
├── mcdm_utils.py         # Utility functions with new analysis features
├── data/
│   └── demo.csv          # Realistic jobs dataset
├── screens/              # Screenshots for documentation
├── README.md
└── LICENSE
How It Works 🧠
1. Data Input Options
Manual Entry: Create custom career scenarios
Realistic Jobs Dataset: 10 real career options (Software Engineer, Investment Banker, Teacher, etc.)
Demo Scenarios: Quick-start examples
2. AHP Pairwise Comparisons
Compare criteria importance (Income vs Hours vs Stability)
Real-time consistency ratio calculation and feedback
Automatic weight normalization using principal eigenvector
3. Behavioral Economics Integration
Time Discounting: Apply exponential decay to model temporal preferences
Short-term bias (high discount rate): Emphasizes immediate benefits like current income
Long-term focus (low discount rate): Weights future-oriented criteria like stability
4. Advanced Analysis
TOPSIS Ranking: Similarity to ideal solution methodology
Intelligent Insights: Explains why options ranked as they did
Performance Analysis: Shows how top choice aligns with your priorities
Screenshots 📸
Default Configuration
Balanced preferences showing moderate time discounting

Realistic Jobs Dataset
10 actual career options with market-based data

High Discount Rate Analysis
Short-term focused results (discount_rate=0.8)

Sensitivity Analysis
Impact of boosting income weight by 30%

Methodology 📚
Academic Foundation
AHP: Saaty, T.L. (1987). The Analytic Hierarchy Process
TOPSIS: Hwang, C.L. & Yoon, K. (1981). Multiple Attribute Decision Making
Validation: Yavuz (2016) - Hybrid fuzzy AHP-TOPSIS for job selection
Novel Integration
Time Discounting: Exponential decay function modeling temporal preferences
Behavioral Economics: Based on research showing people discount future benefits
Generational Insights: Incorporates findings that Gen Z/Millennials prioritize work-life balance
Research Context 🔬
Modern career research shows:

56% of burnout caused by negative work culture, leading to 50% of turnover
Non-wage factors (flexibility, meaning, security) strongly influence life satisfaction (OECD)
Generational shifts: Younger workers prioritize purpose and balance over just compensation
Consistency Guidelines 📏
CR < 0.1: ✅ Good consistency (proceed with confidence)
CR 0.1-0.2: ⚠️ Borderline (consider adjusting comparisons)
CR > 0.2: ❌ Poor consistency (revise judgments for better reliability)
Future Enhancements 🔮
Fuzzy Logic: Handle uncertainty in judgments
Machine Learning: Predict satisfaction from historical data
Group Decisions: Multi-stakeholder career planning
Extended Criteria: Health impact, growth potential, social impact
Monte Carlo: Robust sensitivity analysis under uncertainty
Contributing 🤝
Contributions welcome! Areas of interest:

Additional MCDM methods (PROMETHEE, ELECTRE)
Real-time job market data integration
Mobile-responsive design improvements
Multilingual support
License 📄
MIT License - see LICENSE file for details.

Citation 📖
bibtex
@software{career_decision_analyzer_2025,
  title={Career Decision Analyzer: Multi-criteria Decision Analysis with Time Discounting},
  author={[Your Name]},
  year={2025},
  note={Based on Saaty 1987 (AHP), Hwang \& Yoon 1981 (TOPSIS), and behavioral economics research}
}
