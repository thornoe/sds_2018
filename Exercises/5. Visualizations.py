import warnings

# Other packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# matplotlib inline
sns.set(style="ticks", palette="RdBu")
# for item in ax.get_yticklabels()+ax.get_xticklabels():
#     item.set_fontsize(12)

# Data
iris = sns.load_dataset("iris")
tips = sns.load_dataset("tips")
titanic = sns.load_dataset("titanic")

tips.head(3)
len(tips)
x = tips.total_bill  # tip + total_bill
cuts = np.arange(0, 70, 10)
pd.cut(x, cuts).value_counts()

#############################################################
#   Matplotlib: Univariate distribution using value counts  #
#############################################################
# Changing defaults
plt.style.use("default")  # set style (colors, background, size, gridlines etc.)
plt.rcParams["figure.figsize"] = 6, 4  # set default size of plots
plt.rcParams.update({"font.size": 10})

# Plotting
fig, ax = plt.subplots(figsize=(6, 4))  # Placeholder for plot
# ax contains most of the chart elements: the grid axes, labels, shapes we draw etc.
# fig the actual plot which is displayed (export to pdf etc.)
ax.set_xlim([0, 60])  # x-axis cutoffs
ax.set_ylim([0, 80])  # y-axis cutoffs
ax.hist(x)
fig

#############################################################
#                   Plotting with Pandas                    #
#############################################################
x.head()
x.plot.hist()
#############################################################
#                  Plotting with Seaborn                    #
#############################################################
sns.distplot(x)
# Warning: Kernel densities are great for anonymizing, the line isn't!
# Cumlulative plots:
sns.distplot(x, hist_kws={"cumulative": True}, kde_kws={"cumulative": True})
# Univariate categorical
sns.countplot(x="sex", data=tips)
# Plotting DataFrames w. table format (tidy/long table)
# Two numeric variables
plt.scatter(x=tips["total_bill"], y=tips["tip"])
# Interpolating the data
sns.jointplot(x="total_bill", y="tip", data=tips, kind="hex", size=5)  # hex
# Linear model plot
sns.lmplot(x="total_bill", y="tip", data=tips, size=2.5, aspect=2)
# Barplot (mean for each categorical variable)
f = sns.barplot(x="day", y="tip", data=tips)
# Boxplot (compute quartiles for each categorical variable)
f = sns.boxplot(x="sex", y="tip", data=tips)
# Plot grids (relationship for more than two variables)
sns.pairplot(tips, size=1.5, aspect=1.6)
# FacetGrid (investigate heterogeneous relationships)
g = sns.FacetGrid(tips)
g = g.map(sns.regplot, "total_bill", "tip")
# To smoke or not to smoke (distinctive slopes)
g = sns.FacetGrid(tips, col="smoker")  #  split into two plots
g = g.map(sns.regplot, "total_bill", "tip")
# Histogram in Seaborn
warnings.filterwarnings("ignore")

histplot, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.distplot(sns.load_dataset("tips").total_bill, kde=False, ax=ax)
ax.set_title("Distribution of total bill")
ax.set_xlabel("Total bill, $")
ax.title.set_fontsize(20)
ax.xaxis.label.set_fontsize(16)

#################################
#   Ex. 5.1: Python Plotting    #
#################################
# 5.1.3 combine two figures into a two-panel fugure
a, axes = plt.subplots(1, 2)  # axes is an array with each subplot
# ex. 5.1.1 barplot
titanic.head(3)
sns.barplot(x="pclass", y="survived", data=titanic, hue="sex", ax=axes[0])
# BONUS: Boxplot for the fare-prices within each passenger class

# 5.1.2 scatterplot
iris.head(3)
sns.regplot("sepal_length", "petal_length", data=iris, order=2, ax=axes[1])
b.set_title("Irises, 2nd order polynomial fit")
b.set_xlabel("Sepal length")
b.set_ylabel("Petal length")
# Ex. 5.1.4
plt.show()
# 5.1.4
sns.pairplot(
    iris,
    hue="species",
    vars=["petal_length", "sepal_length"],
    diag_kind="kde",
    plot_kws=dict(s=50, linewidth=1),
    markers="+",
)  # markers set to circles, squares and diamonds: ['o', 's', 'D'], drop plot_kws then
#####################################
#   Ex. 5.2: Eplanatory Plotting    #
#####################################
f = sns.boxplot(x="sex", y="tip", data=tips)


help(sns.set)
