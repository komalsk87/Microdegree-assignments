from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

# 1. Load raw dataset
base_dir = Path(__file__).resolve().parent
raw_file = base_dir / "customer_support_tickets.csv"
clean_file = base_dir / "customer_support_tickets_cleaned.csv"

df = pd.read_csv(raw_file)

# 2. Fill missing values based on data type
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna("No data available")

# 3. Save the cleaned DataFrame to a new CSV file
df.to_csv(clean_file, index=False)
print(f"Cleaned dataset successfully saved to '{clean_file.name}'!")

# 4. Plot distribution of tickets by customer gender
plt.figure(figsize=(8, 5))
ax = sns.countplot(
    data=df,
    x="Customer Gender",
    hue="Customer Gender",
    dodge=False,
    order=df["Customer Gender"].value_counts().index,
    palette="Set1",
    legend=False,
)

plt.title("Distribution of Customer Support Tickets by Gender", fontsize=14)
plt.xlabel("Customer Gender", fontsize=12)
plt.ylabel("Total Count", fontsize=12)
plt.xticks(rotation=0)

# Add counts above bars
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 6),
        textcoords="offset points",
    )

plt.tight_layout()
output_plot = base_dir / "customer_gender_distribution.png"
plt.savefig(output_plot, dpi=150, bbox_inches="tight")

# Only show the plot if a GUI backend is available
if plt.get_backend().lower() != "agg":
    plt.show()

plt.close()
print(f"Chart saved to '{output_plot.name}'")
