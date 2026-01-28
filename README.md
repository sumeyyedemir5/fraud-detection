# 🛡️ Real-Time Credit Card Fraud Detection with MLOps Pipeline

This project is an end-to-end Machine Learning & MLOps solution designed to detect fraudulent banking transactions in real time.
It demonstrates a modern engineering workflow covering everything from data preprocessing and model training to versioning and serverless cloud deployment.

## 🚀 Key Features

**Data Imbalance Handling :**  Optimized XGBoost model using scale_pos_weight to address highly imbalanced data (only 0.17% fraud rate).

**MLOps Versioning:**
Full tracking of datasets and model artifacts with DVC (Data Version Control) for reproducible pipelines.

**API Integration:**
High-performance prediction service built with FastAPI and Pydantic for strict type validation.

**Containerization:**
Fully Dockerized application ensuring consistency across Development, Staging, and Production environments.

**Cloud Deployment:**
Scalable, serverless deployment on Google Cloud Run (GCP).

## 📊 Model Performance

Evaluated on the test dataset:


```
Metric	          Value
F1-Score(Class 1)	0.82
Recall	            0.84
Precision	        0.80
Accuracy	        1.00
```


The model successfully captures 84% of fraudulent transactions while maintaining high precision, minimizing false positives and avoiding inconvenience to legitimate customers.

## 🛠️ Technology Stack

* Language: Python 3.12

* ML Frameworks: XGBoost, Scikit-learn, Pandas, NumPy

* MLOps: DVC, Git

* API & Deployment: FastAPI, Uvicorn, Docker, Google Cloud Run (GCP)

* Monitoring: Evidently AI (planned)

## 📁 Project Structure


```
├── data/               # Raw and processed data (Tracked by DVC)
├── models/             # Trained model artifacts (.pkl)
├── analysis.py         # Exploratory Data Analysis (EDA)
├── preprocess.py       # Data cleaning and scaling
├── train.py            # Model training and evaluation
├── main.py             # FastAPI service script
├── Dockerfile          # Container configuration
├── dvc.yaml            # DVC pipeline definitions
└── requirements.txt    # List of dependencies

```




## ⛓️ DVC Pipeline Explanation

The project utilizes DVC (Data Version Control) to build a fully reproducible machine learning pipeline.
This ensures that every step—from raw data ingestion to the final trained model—is tracked and can be reproduced with a single command.

The pipeline consists of two main stages defined in dvc.yaml:

Preprocess

* Cleans raw credit card transaction data

* Handles missing values

* Performs feature scaling

Dependencies

* data/raw/creditcard.csv

* preprocess.py

Outputs

* data/processed/X_train.csv

* data/processed/y_train.csv


Train

* Trains an XGBoost classifier

* Applies class weight optimization to address extreme data imbalance

Dependencies

* Processed dataset files

* train.py

Outputs

* models/fraud_model.pkl

### Reproduce the Entire Pipeline
```
dvc repro
```

This command automatically executes all stages and regenerates the final model artifact.

##☁️ Cloud Infrastructure & Deployment (GCP)

The model is deployed as a serverless microservice on Google Cloud Platform, chosen for its scalability and cost efficiency in production.

**Deployment Workflow**

* Container Registry: Docker image stored in Google Artifact Registry

* Serverless Execution: Google Cloud Run hosts the FastAPI application

* Authentication: Secure access management via Google Cloud IAM

**Deployment Commands**
```
# 1. Create Artifact Registry Repository
gcloud artifacts repositories create fraud-repo \
  --repository-format=docker \
  --location=europe-west1

# 2. Tag and Push Docker Image
docker tag fraud-api europe-west1-docker.pkg.dev/[PROJECT_ID]/fraud-repo/fraud-api:v1
docker push europe-west1-docker.pkg.dev/[PROJECT_ID]/fraud-repo/fraud-api:v1

# 3. Deploy to Cloud Run
gcloud run deploy fraud-service \
  --image europe-west1-docker.pkg.dev/[PROJECT_ID]/fraud-repo/fraud-api:v1 \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

## 📈 Metrics Evaluation

Given the highly imbalanced nature of fraud detection, optimization focused on the F1-score, balancing precision and recall:

```
Classification Report
Metric	         Class 0 (Normal)	     Class 1 (Fraud)
Precision	      1.00	                  0.80
Recall	          1.00	                  0.84
F1-Score	      1.00	                  0.82
```

This demonstrates strong fraud detection capability while keeping false positives low.

## ⚙️ Installation and Usage
1. Clone the Repository
```
git clone https://github.com/sumeyyedemir5/fraud-detection-mlops.git
cd fraud-detection-mlops
```
2. Install Dependencies
```
pip install -r requirements.txt
```

3. Run the API via Docker
```
docker build -t fraud-api .
docker run -p 8080:80 fraud-api
```
## 🌐 API Usage

Once the API is running, open interactive Swagger documentation:

http://localhost:8080/docs

Example Request (POST /predict)
```
{
  "V1": -1.3598,
  "V2": -0.0727,
  "...": "...",
  "scaled_amount": 0.244
}
```
