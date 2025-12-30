import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv(r'C:\Users\devis\Documents\Osasis\Unemployment in India.csv')
print(df.head())

df= df.dropna(how='all')#removes the row where all the column values in that row are null
df.columns= df.columns.str.strip()
text_cols= df.select_dtypes(include=['object']).columns
for col in text_cols:
    df[col]= df[col].str.strip()
df['Date']= pd.to_datetime(df['Date'],format='%d-%m-%Y')

#March 1st is used for because it separates the Pre-COVID months from the Lockdown months
before_lockdown= df['Date']<'2020-03-01'
during_lockdown= df['Date']>='2020-03-01'
avg_before_lockdown= df.loc[before_lockdown,'Estimated Labour Participation Rate (%)'].mean()
avg_during_lockdown= df.loc[during_lockdown,'Estimated Labour Participation Rate (%)'].mean()

print(f'Avg Unemployment before_lockdown: {avg_before_lockdown:.2f}%')
print(f'Avg Unemployment during_kockdown: {avg_during_lockdown:.2f}%')
print(f'Total Unemployment: {avg_during_lockdown - avg_before_lockdown :.2f}%')

sns.lineplot(data=df, x='Date', y='Estimated Labour Participation Rate (%)', hue='Area')
#March 25th is used here for the line because it was the official start of the national lockdown
plt.axvline(pd.to_datetime('2020-03-25'), color='red', linestyle='--', label='National Lockdown')
plt.title('Impact of COVID-19 Lockdown on Unemployment in India')
plt.xlabel('Timeline')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.show()

state= df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=state, x='Estimated Unemployment Rate (%)', y='Region')
plt.title('Average Unemployment Rate by State')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title('Correlation between Economic')
plt.show()