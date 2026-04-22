# Auto Churn Predictor 🔮

A complete, professional Machine Learning Web Application designed to predict customer churn. Users can upload their own CSV datasets, automatically preprocess the data, train multiple machine learning models, and use an interactive dashboard to evaluate results and make new predictions.

## 🌟 Features

- **Upload Dataset**: Simple drag-and-drop interface for CSV files.
- **Auto Preprocessing**: Automatically removes duplicates, handles missing values (mean for numerical, mode for categorical), and one-hot encodes categorical variables.
- **Multi-Model Training**: Trains and compares **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers simultaneously.
- **Performance Dashboard**: Interactive visualizations built with Plotly, including Accuracy comparisons, Target Class distribution, Feature Importances, and a Correlation Heatmap.
- **Dynamic Predictions**: Automatically generates a prediction form based on your dataset's specific columns to predict churn for new customers.
- **Modern UI/UX**: Premium, responsive interface built entirely with HTML and Vanilla CSS. Features a dynamic Dark Mode toggle!
- **Downloadable Assets**: Download your cleaned dataset and the highest-performing trained model (`.pkl`).

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Visualizations**: Plotly
- **Frontend**: HTML5, Vanilla CSS3, JavaScript
- **Icons**: FontAwesome

## 📂 Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                   # Main Flask application and ML logic
├── requirements.txt         # Python dependencies
├── models/                  # Directory where trained models and scalers are saved
├── uploads/                 # Directory where user datasets are stored
│
├── templates/               # HTML Templates
│   ├── base.html            # Base layout with sidebar and theme toggle
│   ├── index.html           # Landing page
│   ├── upload.html          # Dataset upload page
│   ├── preprocessing.html   # Data preview and target selection
│   ├── training.html        # Model evaluation metrics and comparison
│   ├── dashboard.html       # Interactive Plotly charts
│   └── prediction.html      # Dynamic form for single-user predictions
│
└── static/
    ├── css/
    │   └── style.css        # Premium custom CSS (Dark/Light mode)
    └── js/
        └── script.js        # Frontend interactions and theme management
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pavana3070/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Install the dependencies:**
   Make sure you have Python installed. It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access the Web App:**
   Open your web browser and navigate to: `http://127.0.0.1:5000`

## 💡 How to Use

1. Navigate to the **Upload** page and upload your historical customer data (CSV format).
2. On the **Preprocessing** page, review your cleaned dataset and select your target column (e.g., `Churn`).
3. Click **Train Models**. The app will evaluate multiple algorithms and highlight the best one.
4. Visit the **Dashboard** to explore interactive charts explaining model performance and feature correlations.
5. Go to the **Predict** page, enter a new customer's details, and instantly see their likelihood of churning!

## 📝 License
This project is open-source and available under the MIT License.
