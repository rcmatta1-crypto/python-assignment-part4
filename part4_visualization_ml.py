# Task 1: Data Exploration with Pandas

import pandas as pd

# Load dataset
df = pd.read_csv("students.csv")

# 1. First 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# 2. Shape and data types
print("\n--- Shape ---")
print(df.shape)

print("\n--- Data Types ---")
print(df.dtypes)

# 3. Summary statistics
print("\n--- Summary Statistics ---")
print(df.describe())

# 4. Count of passed vs failed
print("\n--- Pass/Fail Count ---")
print(df['passed'].value_counts())

# 5. Average score per subject for pass and fail
subject_cols = ['math', 'science', 'english', 'history', 'pe']

pass_avg = df[df['passed'] == 1][subject_cols].mean()
fail_avg = df[df['passed'] == 0][subject_cols].mean()

print("\n--- Average Scores (Passed Students) ---")
print(pass_avg)

print("\n--- Average Scores (Failed Students) ---")
print(fail_avg)

# 6. Student with highest overall average
df['average'] = df[subject_cols].mean(axis=1)

top_student = df.loc[df['average'].idxmax()]

print("\n--- Top Student ---")
print(top_student[['name', 'average']])



# Task 2: Data Visualization with Matplotlib

import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("students.csv")

# Add avg_score column
subject_cols = ['math', 'science', 'english', 'history', 'pe']
df['avg_score'] = df[subject_cols].mean(axis=1)


# ---------------- Plot 1: Bar Chart ----------------
avg_scores = df[subject_cols].mean()

plt.figure()
plt.bar(subject_cols, avg_scores)
plt.title("Average Score per Subject")
plt.xlabel("Subjects")
plt.ylabel("Average Score")
plt.savefig("plot1_bar.png")
plt.show()


# ---------------- Plot 2: Histogram ----------------
plt.figure()
plt.hist(df['math'], bins=5)

mean_math = df['math'].mean()
plt.axvline(mean_math, linestyle='--', label=f"Mean: {mean_math:.2f}")

plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("plot2_hist.png")
plt.show()


# ---------------- Plot 3: Scatter Plot ----------------
pass_df = df[df['passed'] == 1]
fail_df = df[df['passed'] == 0]

plt.figure()
plt.scatter(pass_df['study_hours_per_day'], pass_df['avg_score'], label="Pass")
plt.scatter(fail_df['study_hours_per_day'], fail_df['avg_score'], label="Fail")

plt.title("Study Hours vs Average Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Average Score")
plt.legend()
plt.savefig("plot3_scatter.png")
plt.show()


# ---------------- Plot 4: Box Plot ----------------
pass_attendance = df[df['passed'] == 1]['attendance_pct'].tolist()
fail_attendance = df[df['passed'] == 0]['attendance_pct'].tolist()

plt.figure()
plt.boxplot([pass_attendance, fail_attendance], labels=['Pass', 'Fail'])

plt.title("Attendance Distribution (Pass vs Fail)")
plt.ylabel("Attendance Percentage")
plt.savefig("plot4_box.png")
plt.show()


# ---------------- Plot 5: Line Plot ----------------
plt.figure()

plt.plot(df['name'], df['math'], marker='o', label="Math")
plt.plot(df['name'], df['science'], marker='s', label="Science")

plt.xticks(rotation=45)
plt.title("Math and Science Scores by Student")
plt.xlabel("Student Name")
plt.ylabel("Score")
plt.legend()
plt.savefig("plot5_line.png")
plt.show()



# Task 3: Data Visualization with Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("students.csv")

# Add avg_score column
subject_cols = ['math', 'science', 'english', 'history', 'pe']
df['avg_score'] = df[subject_cols].mean(axis=1)


# ---------------- Plot 1: Bar Plot ----------------
plt.figure(figsize=(10, 5))

# Math plot
plt.subplot(1, 2, 1)
sns.barplot(data=df, x='passed', y='math')
plt.title("Average Math Score (Pass vs Fail)")
plt.xlabel("Passed (1=Yes, 0=No)")
plt.ylabel("Math Score")

# Science plot
plt.subplot(1, 2, 2)
sns.barplot(data=df, x='passed', y='science')
plt.title("Average Science Score (Pass vs Fail)")
plt.xlabel("Passed (1=Yes, 0=No)")
plt.ylabel("Science Score")

plt.tight_layout()
plt.savefig("plot6_seaborn_bar.png")
plt.show()


# ---------------- Plot 2: Scatter + Regression ----------------
plt.figure()

# Scatter plot
sns.scatterplot(data=df, x='attendance_pct', y='avg_score', hue='passed')

# Regression lines
sns.regplot(data=df[df['passed']==1], x='attendance_pct', y='avg_score', scatter=False, label='Pass')
sns.regplot(data=df[df['passed']==0], x='attendance_pct', y='avg_score', scatter=False, label='Fail')

plt.title("Attendance vs Average Score")
plt.xlabel("Attendance %")
plt.ylabel("Average Score")
plt.legend()

plt.savefig("plot7_seaborn_scatter.png")
plt.show()


# ---------------- Comments ----------------
# Seaborn makes visualization easier and more visually appealing compared to Matplotlib.
# It automatically handles grouping (like 'passed') and styling with less code.
# However, Matplotlib gives more control for customization, while Seaborn is quicker for statistical plots.




# Task 4: Machine Learning with scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
df = pd.read_csv("students.csv")

# ---------------- Step 1: Prepare Data ----------------
feature_cols = ['math', 'science', 'english', 'history', 'pe', 'attendance_pct', 'study_hours_per_day']
X = df[feature_cols]
y = df['passed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------- Step 2: Train Model ----------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
print("\nTraining Accuracy:", round(train_acc * 100, 2), "%")


# ---------------- Step 3: Evaluate Model ----------------
y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(test_acc * 100, 2), "%")

print("\n--- Predictions ---")
names = df.loc[X_test.index, 'name']

for name, actual, pred in zip(names, y_test, y_pred):
    result = "✅ Correct" if actual == pred else "❌ Wrong"
    print(f"{name} | Actual: {actual} | Predicted: {pred} → {result}")


# ---------------- Step 4: Feature Importance ----------------
coefficients = model.coef_[0]

feature_importance = list(zip(feature_cols, coefficients))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n--- Feature Importance ---")
for feature, coef in feature_importance:
    print(f"{feature}: {coef:.4f}")

# Plot
features = [f[0] for f in feature_importance]
values = [f[1] for f in feature_importance]

colors = ['green' if v > 0 else 'red' for v in values]

plt.figure()
plt.barh(features, values, color=colors)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.savefig("plot8_feature_importance.png")
plt.show()


# ---------------- Step 5: New Student Prediction ----------------
new_student = [[75, 70, 68, 65, 80, 82, 3.2]]

new_scaled = scaler.transform(new_student)

prediction = model.predict(new_scaled)[0]
probability = model.predict_proba(new_scaled)[0]

result = "Pass" if prediction == 1 else "Fail"

print("\n--- New Student Prediction ---")
print("Prediction:", result)
print(f"Probability (Fail, Pass): {probability}")



