import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class ChurnModel:
    """
    Encapsulates the customer churn prediction model pipeline, including data loading,
    preprocessing, training, and inference.
    """

    def __init__(self, csv_path=None):
        """
        Initialize the ChurnModel by loading data and training the classifier.

        :param csv_path: Optional path to the churn CSV dataset. If not provided,
                         uses the default path in the user's Downloads folder.
        :raises FileNotFoundError: If the specified CSV file does not exist.
        """
        # Default location for the dataset if no path is provided
        default_path = r"C:\Users\darsh\Downloads\Churn_Modelling.csv"
        self.csv_path = csv_path or default_path

        # Verify that the CSV file exists
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        # DataFrame to hold the loaded data
        self.df = None
        # Scikit-learn Random Forest classifier instance
        self.model = None
        # Scaler for normalizing numeric features
        self.scaler = None
        # Dictionary to store fitted LabelEncoders for categorical features
        self.label_encoders = {}
        
        # List of features to use for model training and inference
        self.features = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary'
        ]

        # Load the dataset and train the model
        self._load_and_train()

    def _load_and_train(self):
        """
        Internal method to load the dataset, preprocess features, train-test split,
        scale numeric data, and fit the Random Forest classifier.
        Prints test set accuracy upon completion.
        """
        # Load the CSV into a pandas DataFrame
        self.df = pd.read_csv(self.csv_path)

        # Prepare feature matrix X and target vector y
        X = self.df[self.features].copy()
        y = self.df['Exited']

        # Encode categorical columns using LabelEncoder
        for col in ['Geography', 'Gender']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # Split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize StandardScaler and scale numeric columns
        self.scaler = StandardScaler()
        numeric_cols = [
            'CreditScore', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'EstimatedSalary'
        ]
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        # Instantiate and train the Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model performance on the test set
        accuracy = self.model.score(X_test, y_test)
        print(f'Model trained. Test set accuracy: {accuracy:.2f}')

    def predict(self, customer_data):
        """
        Predict whether a single customer will churn based on input features.

        :param customer_data: Dictionary mapping each feature name to its value.
        :return: Boolean indicating churn (True) or retention (False).
        """
        # Convert input data to DataFrame for preprocessing
        df = pd.DataFrame([customer_data])

        # Apply LabelEncoder transformations to categorical fields
        for col in ['Geography', 'Gender']:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])

        # Scale numeric columns using the previously fitted scaler
        numeric_cols = [
            'CreditScore', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'EstimatedSalary'
        ]
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Generate prediction (0 = stay, 1 = churn)
        prediction = self.model.predict(df)
        return bool(prediction[0])


class ChurnGUI:
    """
    Graphical User Interface for loading the churn dataset, entering customer data,
    and displaying churn prediction results.
    """

    def __init__(self, root):
        """
        Initialize the main application window and widgets.

        :param root: Root Tkinter window instance.
        """
        self.root = root
        self.root.title('Customer Churn Prediction')
        self._create_widgets()

    def _create_widgets(self):
        """
        Create and layout the GUI components: load button, dynamic input fields,
        and predict button.
        """
        row = 0
        self.entries = {}

        # Button to load the dataset and train the model
        ttk.Button(
            self.root, text='Load Dataset', command=self._load_dataset
        ).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # Frame to hold feature input widgets dynamically
        self.input_frame = ttk.Frame(self.root)
        self.input_frame.grid(row=row, column=0, columnspan=2)
        row += 1

        # Predict button (disabled until model is loaded)
        self.predict_btn = ttk.Button(
            self.root, text='Predict Churn', command=self._on_predict, state='disabled'
        )
        self.predict_btn.grid(row=row, column=0, columnspan=2, pady=10)

    def _load_dataset(self):
        """
        Prompt the user to select a CSV file, initialize the ChurnModel,
        and build the input interface for each feature.
        """
        path = filedialog.askopenfilename(
            title='Select Churn CSV',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        try:
            self.cm = ChurnModel(csv_path=path)
        except FileNotFoundError as e:
            messagebox.showerror('Error', str(e))
            return

        # Clear any existing inputs
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        # Dynamically create input widgets based on model features
        row = 0
        for feature in self.cm.features:
            ttk.Label(
                self.input_frame, text=f'{feature}:'
            ).grid(row=row, column=0, sticky='W', padx=5, pady=5)

            if feature in ['Geography', 'Gender']:
                # Dropdown for categorical features
                values = sorted(self.cm.df[feature].unique().tolist())
                combo = ttk.Combobox(
                    self.input_frame, values=values, state='readonly'
                )
                combo.current(0)
                combo.grid(row=row, column=1, padx=5, pady=5)
                self.entries[feature] = combo

            elif feature in ['HasCrCard', 'IsActiveMember']:
                # Checkbox for binary features
                var = tk.IntVar()
                chk = ttk.Checkbutton(self.input_frame, variable=var)
                chk.grid(row=row, column=1, padx=5, pady=5)
                self.entries[feature] = var

            else:
                # Text entry for numeric features
                ent = ttk.Entry(self.input_frame)
                ent.grid(row=row, column=1, padx=5, pady=5)
                self.entries[feature] = ent

            row += 1

        # Enable the predict button now that model is ready
        self.predict_btn.config(state='normal')

    def _on_predict(self):
        """
        Gather input values, call the ChurnModel.predict method,
        and display the result in a popup message box.
        """
        try:
            data = {}
            # Extract values from each input widget
            for feature, widget in self.entries.items():
                if isinstance(widget, ttk.Combobox):
                    data[feature] = widget.get()
                elif isinstance(widget, tk.IntVar):
                    data[feature] = widget.get()
                else:
                    data[feature] = float(widget.get())

            # Predict churn (True means will churn)
            churn = self.cm.predict(customer_data=data)
            msg = (
                'Customer is likely to leave.' if churn else 'Customer is likely to stay.'
            )
            messagebox.showinfo('Prediction', msg)

        except Exception as e:
            # Handle invalid or missing input values
            messagebox.showerror('Error', f'Invalid input: {e}')


if __name__ == '__main__':
    # Launch the Tkinter application
    root = tk.Tk()
    app = ChurnGUI(root)
    root.mainloop()
