# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (using an example dataset: Titanic from seaborn library)
sns.get_dataset_names()
df = sns.load_dataset('titanic')

# Display the first 5 rows of the dataset
print(df.head())

# Data cleaning and preprocessing
# Check for missing values
print(df.isnull().sum())

# Fill missing values for 'age' with the median age
df['age'].fillna(df['age'].median(), inplace=True)

# Drop unnecessary columns
df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male'], inplace=True)

# Perform exploratory data analysis (EDA)
# Summary statistics
print(df.describe())

# Visualize the data
# Survival count by gender
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival Count by Gender")
plt.show()

# Age distribution of passengers
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

# Survival rate by passenger class
sns.barplot(x='class', y='survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

# Pair plot to visualize relationships between variables
sns.pairplot(df, hue='survived')
plt.show()

# Heatmap to visualize correlation between variables
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Conclusions
# Based on the visualizations, we can conclude that:
# 1. Female passengers had a higher survival rate than male passengers.
# 2. Passengers in the first class had a higher survival rate compared to other classes.
# 3. Age and fare seem to have a weak correlation with survival rate.
