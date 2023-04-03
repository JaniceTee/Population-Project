# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

df = pd.read_csv(r"C:\Users\princ\Downloads\census-historic-population-borough_London (1).csv")

# EXPLORATORY DATA ANALYSIS
forecasted_growth = df['growth'].value_counts()
print('Forecasted Growth:', '\n', forecasted_growth)
sns.countplot(x='growth', data=df, palette='hls')
plt.title('Forecasted Growth: Up(1)/Down(0)')
plt.show()

# Ratio of Forecasted Growth (Up/Down)
count_downward_growth = len(df[df['growth'] == 0])
count_upward_growth = len(df[df['growth'] == 1])
pct_of_downward_growth = count_downward_growth/(count_downward_growth+count_upward_growth)
print('\n', "percentage of boroughs experiencing a decline in population change is", pct_of_downward_growth*100)
pct_of_upward_growth = count_upward_growth/(count_downward_growth+count_upward_growth)
print("percentage of boroughs experiencing an upward growth in population change is", pct_of_upward_growth*100)

# Histogram for 1991 census.
df.Pop_1991.hist()
plt.title('Histogram of 1991 Census')
plt.xlabel('1991 Population')
plt.ylabel('Frequency')
plt.show()

# Histogram for 2001 census.
df.Pop_2001.hist()
plt.title('Histogram of 2001 Census')
plt.xlabel('2001 Population')
plt.ylabel('Frequency')
plt.show()

# Histogram for 2011 census.
df.Pop_2011.hist()
plt.title('Histogram of 2011 Census')
plt.xlabel('2011 Population')
plt.ylabel('Frequency')
plt.show()

# Histogram for 2021 census.
df.Pop_2021.hist()
plt.title('Histogram of 2021 Census')
plt.xlabel('2021 Population')
plt.ylabel('Frequency')
plt.show()

# Line graph for 1991 census.
plt.plot(df.names, df.Pop_1991, label="1991 Population")
plt.xticks(df.names[::1],  rotation='vertical')
plt.margins(0.2)
plt.title('1991 Census')
plt.xlabel('London Boroughs')
plt.ylabel('Population')
plt.tight_layout()
plt.legend()
plt.show()

# Line graph for 2001 census.
plt.plot(df.names, df.Pop_2001, label="2001 Population")
plt.xticks(df.names[::1],  rotation='vertical')
plt.margins(0.2)
plt.title('2001 Census')
plt.xlabel('London Boroughs')
plt.ylabel('Population')
plt.tight_layout()
plt.legend()
plt.show()

# Line graph for 2011 census.
plt.plot(df.names, df.Pop_2011, label="2011 Population")
plt.xticks(df.names[::1],  rotation='vertical')
plt.margins(0.2)
plt.title('2011 Census')
plt.xlabel('London Boroughs')
plt.ylabel('Population')
plt.tight_layout()
plt.legend()
plt.show()

# Line graph for 2021 census.
plt.plot(df.names, df.Pop_2021, label="2021 Population")
plt.xticks(df.names[::1],  rotation='vertical')
plt.margins(0.2)
plt.title('2021 Census')
plt.xlabel('London Boroughs')
plt.ylabel('Population')
plt.tight_layout()
plt.legend()
plt.show()

# Comparing the population values of the last four census on a single graph.
x_coordinates = df.names
y1_coordinates = df.Pop_1991
y2_coordinates = df.Pop_2001
y3_coordinates = df.Pop_2011
y4_coordinates = df.Pop_2021
plt.plot(x_coordinates, y1_coordinates, label="1991")
plt.plot(x_coordinates, y2_coordinates, label="2001")
plt.plot(x_coordinates, y3_coordinates, label="2011")
plt.plot(x_coordinates, y3_coordinates, label="2021")
plt.xticks(x_coordinates[::1],  rotation='vertical')
plt.margins(0.2)
plt.title('Population of London Boroughs over Last Four Decades')
plt.xlabel('London Boroughs')
plt.ylabel('Population')
plt.tight_layout()
plt.legend()
plt.show()

# Trend Analysis for the City of London only.
x_axis = ['Pop_1801', 'Pop_1811', 'Pop_1821', 'Pop_1831', 'Pop_1841', 'Pop_1851',
          'Pop_1861', 'Pop_1871', 'Pop_1881', 'Pop_1891', 'Pop_1901', 'Pop_1911',
          'Pop_1921', 'Pop_1931', 'Pop_1939', 'Pop_1951', 'Pop_1961', 'Pop_1971',
          'Pop_1981', 'Pop_1991', 'Pop_2001', 'Pop_2011', 'Pop_2021']
y_axis = [129000, 121000, 125000, 123000, 124000, 128000, 112000, 75000, 51000,
          38000, 27000,	20000, 14000, 11000, 9000, 5000, 4767, 4000, 5864, 4230,
          7181,	7375, 8600]
plt.plot(x_axis, y_axis, label="Population Trend")
plt.xticks(x_axis[::1],  rotation='vertical')
plt.margins(0.2)
plt.title('City of London Population Trend Over the Years')
plt.xlabel('Census Years')
plt.ylabel('Population')
plt.tight_layout()
plt.legend()
plt.show()

# Scatter Plot (Correlation between 1991 and 2001 censuses).
gridobj = sns.lmplot(x="Pop_1991", y="Pop_2001", data=df, height=5, aspect=1.6, robust=True, palette='tab10',
                     scatter_kws=dict(s=100, linewidths=.9, edgecolors='black'))
plt.title('Correlation Between Census 1991 and 2001')
plt.show()

# Scatter Plot (Correlation between 2011 and 2021 censuses).
gridobj = sns.lmplot(x="Pop_2011", y="Pop_2021", data=df, height=5, aspect=1.6, robust=True, palette='tab10',
                     scatter_kws=dict(s=100, linewidths=.9, edgecolors='black'))
plt.title('Correlation Between Census 2011 and 2021')
plt.show()

# Detecting Outliers in 1991, 2002, 2011, 2021 censuses.
sns.boxplot(x=df['Pop_1991'])
plt.title('Outliers for 1991 Census')
plt.show()
sns.boxplot(x=df['Pop_2001'])
plt.title('Outliers for 2001 Census')
plt.show()
sns.boxplot(x=df['Pop_2011'])
plt.title('Outliers for 2011 Census')
plt.show()
sns.boxplot(x=df['Pop_2021'])
plt.title('Outliers for 2021 Census')
plt.show()

# Plot correlation matrix and Heatmap.
df = df.drop(['growth'], axis=1)
correlation = df.corr(numeric_only=True)
print('\nCorrelation Matrix:')
print('\n', correlation)
plt.figure(figsize=(10, 5))
sns.heatmap(correlation, cmap="BrBG", annot=True)
plt.tight_layout()
plt.show()
