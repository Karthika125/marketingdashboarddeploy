Marketing Dashboard
Overview

This is an interactive Marketing Dashboard built with Streamlit and Plotly. It analyzes weekly marketing and business data, allowing users to explore campaign performance, media spend, promotions, and revenue.

The dashboard also includes Machine Learning / MMM modeling capabilities to understand drivers of revenue, feature effects, and channel contributions.

Features

📊 Interactive visualizations using Plotly Express and Plotly Graph Objects

🧮 Handles multiple data sources: social media campaigns (Facebook, Google, TikTok, Snapchat), direct response (Email/SMS), promotions, pricing, followers, and revenue

📈 Machine learning-based Revenue modeling (MMM approach)

🔍 Filters and metrics for channel comparison, spend analysis, and ROI insights

🌐 Ready for deployment on Render or other cloud platforms

Folder Structure
marketing_dashboard/           ← Root directory
├── app.py                     ← Main Streamlit app
├── requirements.txt           ← Python dependencies
├── README.md                  ← Project documentation
├── .gitignore                 ← Ignore files for Git
└── data/                      ← Optional folder for CSV / datasets

Installation (Local)

Clone the repository:

git clone https://github.com/username/marketing_dashboard.git
cd marketing_dashboard


(Optional) Create a virtual environment:

python -m venv venv
# Activate:
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py


Open the dashboard at http://localhost:8501.

Deployment on Render

Connect your GitHub repository to Render
.

Set Build Command:

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt


Set Start Command:

streamlit run app.py --server.port $PORT --server.enableCORS false


Render will automatically build and deploy the app, giving you a public URL.

Dependencies

Python 3.11

Streamlit 1.26.0

Plotly 6.3.0

Pandas 2.1.1

Numpy 1.27.0

Scikit-learn 1.3.1

(See requirements.txt for full list)

Notes

Keep requirements.txt updated if new packages are added.

Store large datasets in data/ but consider cloud storage for deployment.

Use $PORT in start command for cloud deployment.

License

MIT License. See LICENSE file for details.