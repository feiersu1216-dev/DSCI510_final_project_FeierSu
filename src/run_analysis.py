# CORRELATION BAR CHART - ONLY NUMERICALS
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

df = pd.read_csv("data/processed/kpop_physical_sales.csv")

numeric_df = df.select_dtypes(include=['float64', 'int64'])

results = []

for col in numeric_df.columns:
    if col == "sales":
        continue
    data = numeric_df[["sales", col]].dropna()
    if len(data) > 2:
        r, p = pearsonr(data["sales"], data[col])
        results.append((col, r, p))
    else:
        results.append((col, np.nan, np.nan))

results_df = pd.DataFrame(results, columns=["Predictor", "Correlation", "P_value"])

results_df["abs_corr"] = results_df["Correlation"].abs()

def sig_color(p):
    if pd.isna(p):
        return "gray"
    elif p < 0.01:
        return "darkgreen"
    elif p < 0.05:
        return "green"
    elif p < 0.1:
        return "lightgreen"
    else:
        return "gray"

results_df["color"] = results_df["P_value"].apply(sig_color)

results_df = results_df.sort_values("abs_corr", ascending=True)

print(results_df.sort_values("abs_corr", ascending=False)[["Predictor", "Correlation", "P_value"]])


# CORRELATION MATRIX
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

df = pd.read_csv("data/processed/kpop_physical_sales.csv")

numeric_df = df.select_dtypes(include=['float64', 'int64'])

cols = numeric_df.columns
n = len(cols)

corr = numeric_df.corr(method='pearson')

pvals = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)

for i in range(n):
    for j in range(n):
        if i == j:
            pvals.iloc[i, j] = 0.0
        else:
            valid = numeric_df[[cols[i], cols[j]]].dropna()
            if len(valid) > 2:
                _, p = pearsonr(valid[cols[i]], valid[cols[j]])
                pvals.iloc[i, j] = p
            else:
                pvals.iloc[i, j] = np.nan

def annot_func(c, p):
    if np.isnan(p):
        return f"{c:.2f}\n(p=NA)"
    else:
        return f"{c:.2f}\n(p={p:.3f})"

annot = np.empty(corr.shape, dtype=object)
for i in range(n):
    for j in range(n):
        annot[i, j] = annot_func(corr.iloc[i, j], pvals.iloc[i, j])

print("Correlation matrix:\n", corr)
print("\nP-values matrix:\n", pvals)


# TOTAL SALES BY PARENT COMPANY
import pandas as pd

df = pd.read_csv("data/processed/kpop_physical_sales.csv")

df_clean = df.dropna(subset=["parent_company", "sales"])

sales_summary = df_clean.groupby("parent_company")["sales"].sum().reset_index()
sales_summary = sales_summary.rename(columns={'sales': 'total_sales'})

sales_summary = sales_summary.sort_values(by="total_sales", ascending=False).reset_index(drop=True)
sales_summary['rank'] = sales_summary.index + 1

print("Total sales by parent company:\n", sales_summary)


# MEDIAN SALES BY PARENT COMPANY
import pandas as pd

df = pd.read_csv("data/processed/kpop_physical_sales.csv")

df_clean = df.dropna(subset=["parent_company", "sales"])

sales_summary = df_clean.groupby("parent_company")["sales"].median().reset_index()
sales_summary = sales_summary.rename(columns={'sales': 'median_sales'})

sales_summary = sales_summary.sort_values(by="median_sales", ascending=False).reset_index(drop=True)
sales_summary['rank'] = sales_summary.index + 1

print("Median sales by parent company:\n", sales_summary)


# ANOVA - SALES BY COMPANY
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("data/processed/kpop_physical_sales.csv")

anova_df = df.dropna(subset=["parent_company", "sales", "annual_awards"])

groups_sales = [group["sales"].values for name, group in anova_df.groupby("parent_company")]
f_sales, p_sales = f_oneway(*groups_sales)
print(f"Sales ANOVA across parent_company: F={f_sales:.3f}, p={p_sales:.3f}")

tukey_sales = pairwise_tukeyhsd(endog=anova_df["sales"], groups=anova_df["parent_company"], alpha=0.05)
print("\nTukey HSD post-hoc results for sales:\n")
print(tukey_sales.summary())


# ANOVA BY GROUP TYPE
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("data/processed/kpop_physical_sales.csv")

anova_df = df.dropna(subset=["group_type", "sales", "annual_awards"])

groups_sales = [group["sales"].values for name, group in anova_df.groupby("group_type")]
f_sales, p_sales = f_oneway(*groups_sales)
print(f"Sales ANOVA across group_type: F={f_sales:.3f}, p={p_sales:.3f}")

tukey_sales = pairwise_tukeyhsd(endog=anova_df["sales"], groups=anova_df["group_type"], alpha=0.05)
print("\nTukey HSD post-hoc results for sales by group_type:\n")
print(tukey_sales.summary())
