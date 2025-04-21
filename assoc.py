import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

# Load dataset
df = pd.read_csv('Groceries_dataset.csv')  
df.head()
df.isna()

df['Transaction'] = df['Member_number'].astype(str) + '_' + df['Date']
transactions = df.groupby('Transaction')['itemDescription'].apply(list).tolist()

# Encode for Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)


from mlxtend.frequent_patterns import apriori
# Generate frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)

from mlxtend.frequent_patterns import association_rules
# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Sort by lift and get top 5
top_rules = rules.sort_values(by='lift', ascending=False).head(5)
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


import matplotlib.pyplot as plt
item_counts = df_encoded.sum().sort_values(ascending=False).head(11)
plt.figure(figsize=(10,6))
item_counts.plot(kind='bar', color='salmon')
plt.title("Top 10 Items by Frequency")
plt.xlabel("Items")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


strong_rule = top_rules.iloc[0]
print(strong_rule)

strong_positive_rules = rules[rules['lift'] > 1].sort_values(by='lift', ascending=False)
print(strong_positive_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
