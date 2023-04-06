# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (using an example dataset: Titanic from seaborn library)
sns.get_dataset_names()
df = sns.load_dataset('titanic')

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

# Visualize the data using subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 16))

# 1. Histogram (Age distribution of passengers)
sns.histplot(df['age'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age Distribution of Passengers")

# 2. Pie chart (Proportion of passengers by class)
class_counts = df['class'].value_counts()
axes[0, 1].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
axes[0, 1].set_title("Proportion of Passengers by Class")

# 3. Bar chart (Survival rate by passenger class)
sns.barplot(x='class', y='survived', data=df, ax=axes[1, 0])
axes[1, 0].set_title("Survival Rate by Passenger Class")

# 4. Scatter plot (Fare vs. Age, colored by survival status)
sns.scatterplot(x='age', y='fare', hue='survived', data=df, ax=axes[1, 1])
axes[1, 1].set_title("Fare vs. Age")

# 5. Survival count by gender
sns.countplot(x='sex', hue='survived', data=df, ax=axes[2, 0])
axes[2, 0].set_title("Survival Count by Gender")

# 6. Box plot (Age distribution by class and survival status)
sns.boxplot(x='class', y='age', hue='survived', data=df, ax=axes[2, 1])
axes[2, 1].set_title("Age Distribution by Class and Survival Status")

# Adjust the layout
plt.tight_layout()

# Display all the visualizations
plt.show()
