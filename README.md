# Customer Churn Prediction App

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [How to Install and Run](#how-to-install-and-run)
- [How to Use the Project](#how-to-use-the-project)
- [Media](#media)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)
- [Author](#author)
- [Connect with Me](#connect-with-me)

---

## Project Description
This desktop application predicts customer churn using a Random Forest classifier. It allows you to load a dataset, enter customer details through an interactive GUI, and receive a churn prediction in real time.

---

## Features
- Load and preprocess customer churn dataset (CSV)
- Label encoding for categorical variables (`Geography`, `Gender`)
- Standard scaling of numeric features
- Train/Test split and model evaluation (80/20 split)
- Random Forest classification with performance reporting
- User-friendly Tkinter-based GUI for data input and prediction

---

## Prerequisites
- Python 3.7 or higher
- `pandas`, `numpy` libraries
- `scikit-learn` for machine learning pipeline
- `tkinter` for GUI components (usually included with Python)

---

## How to Install and Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/DarshanKagi/Customer-Churn-Prediction.git
   cd churn-prediction-app
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**
   ```bash
   python churn_app.py
   ```

---

## How to Use the Project
1. Click **Load Dataset** and select your `Churn_Modelling.csv` file.
2. Fill in each customer feature (e.g., CreditScore, Age, Geography).
3. Click **Predict Churn** to see if the customer is likely to leave.

---

## Media
> Screenshots and demo GIFs coming soon!

---

## Future Enhancements
- Web-based interface using Flask or Django
- Export predictions to CSV or Excel
- Additional model types (e.g., XGBoost, Logistic Regression)
- Model explainability (SHAP, LIME)
- Batch prediction mode for multiple customers

---

## Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a pull request

Please abide by the existing coding style and include tests where applicable.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Technologies
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Tkinter](https://docs.python.org/3/library/tk.html)

---

## Author
**Darshan Kagi**

---

## Connect with Me
[LinkedIn](https://www.linkedin.com/in/darshan-kagi-938836255)

