import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300

df = pd.read_csv('avsurvey2019data.csv')

print("\nColumn Data Types:")
print(df.dtypes)
print("\nSample of first few rows:")
print(df.head())

key_columns = ['SafeAv', 'SafeHuman', 'AvImpact', 'FamiliarityTech', 'FamiliarityNews', 
               'Age', 'BikePghMember', 'SchoolZoneManual', 'Speed25Mph', 'TwoEmployeesAv',
               'SharedCyclist', 'SharedPedestrian', 'ShareTripData', 'SharePerformanceData',
               'AutoOwner', 'SmartphoneOwner', 'ReportSafetyIncident']

df = df.dropna(subset=key_columns)

categorical_cols = ['FamiliarityTech', 'FamiliarityNews', 'BikePghMember', 
                   'SharedCyclist', 'SharedPedestrian', 'AutoOwner', 'SmartphoneOwner']
for col in categorical_cols:
    df[col] = df[col].astype('category')

def convert_age_range(age_range):
    try:
        parts = age_range.split('-')
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
        else:
            return np.nan 
    except:
        return np.nan  

df['Age'] = df['Age'].apply(convert_age_range)
df = df.dropna(subset=['Age']) 

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100],
                        labels=['<25', '25-40', '40-60', '>60'])

def save_plot(fig, filename):
    plt.figure(fig.number) 
    plt.show() 
    fig.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

os.makedirs('plots', exist_ok=True)


fig1 = plt.figure(figsize=(15, 8))
safety_data = pd.melt(df[['SafeAv', 'SafeHuman']], var_name='Vehicle Type', value_name='Safety Score')
sns.violinplot(data=safety_data, x='Vehicle Type', y='Safety Score', inner='box')
plt.title('Safety Score Distribution: AV vs Human-Driven Vehicles')
save_plot(fig1, 'safety_comparison_violin')

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
sns.boxplot(x='AgeGroup', y='SafeAv', data=df, ax=ax1)
ax1.set_title('AV Safety Perception by Age Groups')
sns.boxplot(x='FamiliarityTech', y='SafeAv', data=df, ax=ax2)
ax2.set_title('AV Safety Perception by Tech Familiarity')
save_plot(fig2, 'safety_by_demographics')

fig3 = plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='AvImpact', bins=20, kde=True)
plt.title('Distribution of AV Impact Perception')
save_plot(fig3, 'impact_distribution')

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot(x='FamiliarityTech', y='AvImpact', data=df, ax=ax1)
ax1.set_title('AV Impact by Technical Familiarity')
sns.barplot(x='AgeGroup', y='AvImpact', data=df, ax=ax2)
ax2.set_title('AV Impact by Age Group')
save_plot(fig4, 'impact_by_demographics')

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical columns for correlation:")
print(numerical_cols)

fig5 = plt.figure(figsize=(15, 12))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Variables')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
save_plot(fig5, 'correlation_heatmap')

context_cols = ['SchoolZoneManual', 'Speed25Mph', 'TwoEmployeesAv']
fig6, axes = plt.subplots(1, 3, figsize=(20, 6))
for i, col in enumerate(context_cols):
    sns.countplot(x=col, data=df, ax=axes[i])
    axes[i].set_title(f'Support for {col}')
    axes[i].tick_params(axis='x', rotation=45)
save_plot(fig6, 'context_support')

fig7 = plt.figure(figsize=(15, 10))
g = sns.FacetGrid(df, col='SharedCyclist', row='SharedPedestrian', height=5)
g.map_dataframe(sns.histplot, 'SafeAv')
g.fig.suptitle('Safety Perception by Cyclist/Pedestrian Status', y=1.02)
save_plot(g.fig, 'safety_by_transport_mode')

fig8, axes = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot(x='ShareTripData', hue='BikePghMember', data=df, ax=axes[0,0])
axes[0,0].set_title('Trip Data Sharing by BikePGH Membership')
sns.countplot(x='SharePerformanceData', hue='BikePghMember', data=df, ax=axes[0,1])
axes[0,1].set_title('Performance Data Sharing by BikePGH Membership')
sns.countplot(x='ShareTripData', hue='AutoOwner', data=df, ax=axes[1,0])
axes[1,0].set_title('Trip Data Sharing by Car Ownership')
sns.countplot(x='SharePerformanceData', hue='SmartphoneOwner', data=df, ax=axes[1,1])
axes[1,1].set_title('Performance Data Sharing by Smartphone Ownership')
save_plot(fig8, 'data_sharing_analysis')

fig9 = plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution of Survey Respondents')
save_plot(fig9, 'age_distribution')


fig10, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
sns.countplot(x='FamiliarityTech', data=df, ax=ax1)
ax1.set_title('Distribution of Technical Familiarity')
ax1.tick_params(axis='x', rotation=45)
sns.countplot(x='FamiliarityNews', data=df, ax=ax2)
ax2.set_title('Distribution of News Familiarity')
ax2.tick_params(axis='x', rotation=45)
save_plot(fig10, 'familiarity_distribution')

print("\nStatistical Analysis:")
t_stat, p_value = stats.ttest_rel(df['SafeAv'], df['SafeHuman'])
print(f"\n1. T-test comparing AV and Human safety scores:")
print(f"T-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")

age_groups = df.groupby('AgeGroup')['SafeAv'].apply(list)
f_stat, p_value = stats.f_oneway(*age_groups)
print(f"\n2. ANOVA test for age groups and safety perception:")
print(f"F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}")



print("\nSummary Statistics for Key Variables:")
print(df[['SafeAv', 'SafeHuman', 'AvImpact', 'Age']].describe())

print("\nAll plots have been displayed and saved in the 'plots' directory.")
print("You can find the saved plots at: ./plots/")
