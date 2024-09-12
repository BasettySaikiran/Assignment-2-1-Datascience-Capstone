# Diabetes Prediction Using Artificial Neural Network

# Overview
This Project uses ANN to predict whether a patient is likely to have diabetes based on Pima Indians Diabetes Datset.
The dataset consists of 768 samples With 8 features related to 
health and lifestyle factors contains the following features:
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration
Blood Pressure: Diastolic blood pressure (mm Hg)
Skin Thickness: Triceps skinfold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
Diabetes Pedigree Function: A function that scores the likelihood of diabetes based on family history
Age: Age in years

# Model Architecture
The Artificial Neural Network (ANN) consists of:
Input Layer: 8 features (one for each feature in the dataset).
Hidden Layers: 2 fully connected hidden layers, each with 400 neurons and ReLU activation.
Dropout Layers: Added to each hidden layer to prevent overfitting (dropout rate = 20%).
Output Layer: 1 neuron with a sigmoid activation function for binary classification (diabetes or no diabetes).

# Process of Running the code
Requirements: Python, Tensorflow, Numpy, Pandas, Scikit-Learn, Seaborn, Matplotlib
I have installed the required dependencies, used a jupyter notebook for training the model , accessing the dataset named "diabetes.csv"

In this Project, the binary outcomes for diabetes (i.e., whether a person has diabetes or not) were predicted using the Pima Indians dataset. 
The ongoing project, which aims to predict pre-diabetes, type 2 diabetes, or gestational diabetes using new demographic, lifestyle, and physiological data, can, however, be aligned with the current ANN model through expansion and adaptation. 
In order to do this, we intend to enlarge the dataset by adding more detailed characteristics including dietary preferences, physical activity levels, ethnicity, and particular ailments linked to gestational diabetes. 
Additionally, by changing the output layer to predict the three different forms of diabetes rather than just a binary outcome, the model will be modified for multiclass classification. 
In the end, this improved model will be incorporated into an intuitive application that people with diabetes or healthcare professionals may use to determine their risk for the disease and plan for prompt intervention.
