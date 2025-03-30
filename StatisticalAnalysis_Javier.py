########## 1. Import required libraries ##########
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import ast
import time

########## 2. Define the project and load data ##########

# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'caffe'

svm_df = pd.read_csv(f'{project}_SVM.csv')  # SVM results
baseline_df = pd.read_csv(f'{project}_NB.csv')  # Baseline results

########## 3. Define parameters ##########

# Debugging: Print the input data
print("SVM DataFrame:")
print(svm_df[['CV_list(acc)', 'CV_list(prec)', 'CV_list(rec)', 'CV_list(f1)', 'CV_list(AUC)']])

print("Baseline DataFrame:")
print(baseline_df[['CV_list(acc)', 'CV_list(prec)', 'CV_list(rec)', 'CV_list(f1)', 'CV_list(AUC)']])

# List of metrics to compare
metrics = ['CV_list(acc)', 'CV_list(prec)', 'CV_list(rec)', 'CV_list(f1)', 'CV_list(AUC)']

mean_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'] 

########## 4. Plotting and statistical analysis ##########


##### 4.1 Plotting the Pareto Front for all values of F1 and AUC ####

# Choose two metrics for the axes in Pareto 2D scatter plot (e.g., F1 and AUC)
x_metric = 'CV_list(f1)'
y_metric = 'CV_list(AUC)'

# Extract the full lists of values for the chosen metrics
# These are stored as strings in the CSV, so we need to convert them to lists
# using ast.literal_eval
svm_x_vals = ast.literal_eval(svm_df[x_metric][0])
svm_y_vals = ast.literal_eval(svm_df[y_metric][0])

nb_x_vals = ast.literal_eval(baseline_df[x_metric][0])
nb_y_vals = ast.literal_eval(baseline_df[y_metric][0])

# Plot the Pareto Front
plt.figure(figsize=(8, 6))

# Plot SVM points
plt.scatter(svm_x_vals, svm_y_vals, color='blue', label='SVM', marker='o', alpha=0.7)

# Plot NB points
plt.scatter(nb_x_vals, nb_y_vals, color='orange', label='NB', marker='x', alpha=0.7)

# Add labels and title
plt.xlabel('F1')
plt.ylabel('AUC')
plt.title(f'Pareto Front: F1 vs. AUC ({project})')
plt.legend()
plt.grid(True)

# Save and show the plot
timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f'pareto_front_all_values_{x_metric}_vs_{y_metric}_{timestamp}.png')
plt.show()



##### 4.2 Plotting the mean points for each metric #####

# Calculate mean values for each metric
svm_means = [svm_df[metric][0] for metric in mean_metrics]
baseline_means = [baseline_df[metric][0] for metric in mean_metrics]

# Plot the mean points
plt.figure(figsize=(10, 6))

# Plot SVM mean points
plt.scatter(mean_metrics, svm_means, color='blue', label='SVM Mean', marker='o')

# Plot NB mean points
plt.scatter(mean_metrics, baseline_means, color='orange', label='NB Mean', marker='x')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Mean Value')
plt.title(f'Mean Metric Values for SVM and NB ({project})')
plt.legend()
plt.grid(True)

# Save and show the plot
timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f'mean_points_plot_{timestamp}.png')
plt.show()

##### 4.3 Wilcoxon signed-rank test for each metric #####

results = []

for metric in metrics:
    try:
        # Extract metric values
        svm_vals = ast.literal_eval(svm_df[metric][0])
        baseline_vals = ast.literal_eval(baseline_df[metric][0])

        # Debugging: Print the values
        print(f"Metric: {metric}")
        print(f"SVM values: {svm_vals}")
        print(f"Baseline values: {baseline_vals}")

        # Check for identical data
        if svm_vals == baseline_vals:
            print(f"Warning: Identical data for metric {metric}")

        # Perform Wilcoxon signed-rank test
        wilcoxon_stat, wilcoxon_pval = wilcoxon(svm_vals, baseline_vals, alternative='two-sided')

        # Store results
        results.append({
            'Metric': metric,
            'Wilcoxon p-value': wilcoxon_pval
        })
    except Exception as e:
        print(f"Error processing metric {metric}: {e}")
        results.append({
            'Metric': metric,
            'Wilcoxon p-value': None
        })


# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Replace missing p-values with a placeholder (e.g., 1.0 for plotting purposes)
results_df['Wilcoxon p-value'] = results_df['Wilcoxon p-value'].fillna(1.0)

# Debugging: Print the results DataFrame
print("Results DataFrame:")
print(results_df)

# Plot the p-values
plt.figure(figsize=(10, 6))
x = range(len(metrics))

# Bar plot for Wilcoxon p-values
bars = plt.bar(x, results_df['Wilcoxon p-value'], width=0.4, label='Wilcoxon p-value', color='orange')

# Annotate the bars with the p-values
for bar, pval in zip(bars, results_df['Wilcoxon p-value']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X-coordinate
        bar.get_height() + 0.02,           # Y-coordinate (slightly above the bar)
        f'{pval:.2e}',                     # Format the p-value in scientific notation
        ha='center', va='bottom', fontsize=10
    )

# Add labels and title
plt.xticks(x, metrics, rotation=45)
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
plt.ylabel('P-value')
plt.title('P-values for Wilcoxon Test for the project ' + project)
plt.legend()
plt.tight_layout()

# Save and show the plot with a unique filename
timestamp = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f'p_values_plot_{timestamp}.png')
plt.show()