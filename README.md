<h1>Loan Default Risk Prediction Using XGBoost + SHAP</h1>

<p>A full-stack machine learning project to predict whether a loan will default - using real world financial data, rigorous preprocessing, advanced modeling and interpretability tools like SHAP.</p>

<hr>

<h2>Project Overview</h2>

<ul>
  <li>Binary Classification model: Predict default (1) or Non-default (0)</li>
  <li>Focus on end-to-end ML pipeline with advanced feature engineering</li>
  <li>Model: XGBoost + SHAP explainability</li>
</ul>

<hr>

<h2>Dataset Summary</h2>

<ul>
  <li>Around 50,000 loan records</li>
  <li>Target: <code>default</code> -> 1 (Defaulted), 0 (good)</li>
  <li>Key Features: amount, income, Debt-to-Income ratio, employment, revolving balance, etc.</li>
</ul>

<hr>

<h2>ML Pipeline</h2>

<h3>Feature Engineering</h3>
<ul>
  <li>Ratio features: <code>loan-to-income</code>, <code>installment-to-income</code></li>
  <li>Log transformations: <code>log_income</code>, <code>log_loan_amt</code>, <code>log_revol_bal</code></li>
  <li>State wise risk mapping via <code>state_risk</code></li>
  <li>Binning: <code>income_bin</code></li>
</ul>

<h3>Data Cleaning</h3>
<ul>
  <li>Missing vaue imputation using median / mode / 'unknown'</li>
  <li>Outlier handling via IQR based capping</li>
  <li>Encoding: Ordinal for <code>grade</code>, One-hot for categorical columns</li>
</ul>

<hr>

<h2>Model</h2>

<ul>
  <li><strong>XGBoostClassifier</strong> with RandomizedSearchCV</li>
  <li>Optimized for <strong>Recall (Class 1)</strong> - catching defaulters</li>
  <li>Features scaled using StandardScaler</li>
</ul>

<hr>

<h2>Model Evaluation</h2>

<table>
  <tr><th>Metric</th><th>Score</th></tr>
  <tr><td>Accuracy</td><td>~85%</td></tr>
  <tr><td>Recall (1)</td><td>0.78-0.82</td></tr>
  <tr><td>Precision</td><td>~70%</td></tr>
  <tr><td>AUC-ROC</td><td>0.90+</td></tr>
</table>

<p>Visuals include Confusion Matrix, ROC Curve, Precision-Recall Curve</p>

<hr>

<h2>SHAP Explainability</h2>

<ul>
  <li>SHAP Summary Plot -> Global feature importnace</li>
  <li>Waterfall Plot -> Local prediction breakdowns</li>
  <li>Explains model predictions and clearly and transparently</li>
</ul>

<hr>

<h2>Top Features (XGBoost + SHAP)</h2>

<ul>
  <li>loan_to_income</li>
  <li>installment_to_income</li>
  <li>revolRatio</li>
  <li>log_income</li>
  <li>grade</li>
  <li>term</li>
</ul>

<hr>

<h2>Workflow Summary</h2>

<ol>
  <li>Load and clean data</li>
  <li>Handle missing values and outliers</li>
  <li>Feature Engineering + Encoding</li>
  <li>Scaling + Train-test split</li>
  <li>XGBoost training + Tuning</li>
  <li>Model evaluation + SHAP explainability</li>
</ol>

<hr>

<h2>Repository Contents</h2>

<ul>
  <li><code>FinishedLoanRisk.ipynb</code> -> Full Google Colab Notebook</li>
  <li><code>README.md</code> -> This file</li>
  <li><code>xgb_model.pkl</code> -> Full Google Colab Notebook</li>
</ul>

<hr>

<h2> Requirements</h2>
<p>Install all dependencies at once using:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<h4> Libraries Used:</h4>
<pre><code>pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
</code></pre>

<hr>

<h2>How to Run</h2>

<pre>
  #Clone the repo
  git clone https://github.com/your-username/loan-default-risk-ml.git
  cd loan-default-risk-ml

  # Install packages
  pip install -r requirements.txt

  # Run in Colab or Jupyter Notebook
</pre>

<hr>

<h2>Future Scope</h2>

<ul>
  <li>Deploy via Streamlit or Flask</li>
  <li>Add Model comparison(RF, LightGBM, etc.)</li>
  <li>Build a real time risk scoring tool</li>
</ul>

<hr>

<h2>Author</h2>

<p>
  <b>Aaryan Rathod</b><br>
  CS Undergrad | ML Enthusiast | Future Builder <br>
  🔗<a href="https://github.com/aaryanrathod">GitHub</a>
</p>

<hr>

<h3>Final Words</h3>

<p>
  Built for real world impact.<br>
  Handles dirty data ✅, explains predictions ✅, and tunes smartly for recall ✅.<br>
  If you found this useful — star the repo ⭐ and show some love.
</p>
