# Agri-Project 🌾

**Agri-Project** is a Django-based agricultural assistance system that uses machine learning models to predict soil quality and offer crop suggestions.  
It’s built to help farmers and agri-enthusiasts analyze soil and get actionable insights.

---

## 🚀 Features

✅ Soil quality prediction using trained ML models  
✅ Crop recommendation based on soil features  
✅ Interactive web interface using Django  
✅ Modular and ready for extension  
✅ A solid starter for Agri-tech solutions

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend   | Python (Django) |
| Machine Learning | scikit-learn |
| Templates | Django Templating (HTML) |
| Models & Artifacts | joblib |

---

## 📦 Prerequisites

You need:

* Python **>=3.8**
* `pip` package manager
* Optional: virtual environment tool (recommended)

---

## 🛠️ Installation & Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/Bhaskara-M/agri-project.git
   cd agri-project
   ````

2. **Create and activate a virtual env**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux/macOS
   .\venv\Scripts\activate       # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run database migrations**

   ```bash
   python manage.py migrate
   ```
5. **Start the server**

   ```bash
   python manage.py runserver
   ```
6. **Open in browser**

   ```
   http://127.0.0.1:8000
   ```

---

## 📊 Usage

* Upload soil data or enter values manually.
* The system will use the trained model to predict soil quality.
* Based on the prediction, it provides crop or soil insights.
* Extend by adding more ML models or UI enhancements.

---

## 🧪 Model Files

The following ML artifacts are included and used in the app:

* `scaler.joblib`
* `soil_model.joblib`
* `soil_pipeline.joblib`

These are loaded at runtime to process inputs and make predictions.

---

## 🏗️ Folder Structure

```
agri-project/
├── agri_ai/
├── core/
├── model/
├── templates/
├── manage.py
├── requirements.txt
├── .gitignore
├── README.md
└── *.joblib
```

---

## 🤝 Contributing

1. Fork the repo
2. Create your branch (`git checkout -b feature/foo`)
3. Add your changes
4. Commit (`git commit -m "Add foo feature"`)
5. Push (`git push origin feature/foo`)
6. Open a Pull Request

---

## 📝 License

This project currently **MIT license**.

---
