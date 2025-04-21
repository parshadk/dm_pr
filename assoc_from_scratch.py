import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
# Load dataset (assume CSV with 1 column: 'items')
data = pd.read_csv("groceries.csv", header=None, names=['items'])
transactions = data['items'].apply(lambda x: x.strip().split(','))
# Convert to list of lists
transaction_list = transactions.tolist()


from itertools import combinations
def get_itemset_from_transactions(transactions):
    itemset = set()
    for transaction in transactions:
        for item in transaction:
            itemset.add(frozenset([item]))
    return itemset
def get_frequent_itemsets(transactions, candidates, min_support, freq_itemsets):
    item_count = defaultdict(int)
    for transaction in transactions:
        transaction = set(transaction)
        for candidate in candidates:
            if candidate.issubset(transaction):
                item_count[candidate] += 1

    total_transactions = len(transactions)
    new_freq_itemsets = {}
    for itemset, count in item_count.items():
        support = count / total_transactions
        if support >= min_support:
            new_freq_itemsets[itemset] = support
            freq_itemsets[itemset] = support
    return new_freq_itemsets
def apriori(transactions, min_support):
    itemset = get_itemset_from_transactions(transactions)
    freq_itemsets = {}
    k = 1
    current_freq_itemsets = get_frequent_itemsets(transactions, itemset, min_support, freq_itemsets)

    while current_freq_itemsets:
        k += 1
        candidates = set([i.union(j) for i in current_freq_itemsets for j in current_freq_itemsets if len(i.union(j)) == k])
        current_freq_itemsets = get_frequent_itemsets(transactions, candidates, min_support, freq_itemsets)
    return freq_itemsets


def generate_rules(freq_itemsets, min_confidence):
    rules = []
    for itemset in freq_itemsets:
        if len(itemset) >= 2:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if antecedent in freq_itemsets and consequent:
                        confidence = freq_itemsets[itemset] / freq_itemsets[antecedent]
                        if confidence >= min_confidence:
                            lift = confidence / freq_itemsets[consequent]
                            rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': freq_itemsets[itemset],
                                'confidence': confidence,
                                'lift': lift
                            })
    return sorted(rules, key=lambda x: x['lift'], reverse=True)


min_support = 0.01
min_confidence = 0.3
freq_itemsets = apriori(transaction_list, min_support)
rules = generate_rules(freq_itemsets, min_confidence)
# Show top 5 rules by lift
top_rules = rules[:5]
for i, rule in enumerate(top_rules, 1):
    print(f"{i}. Rule: {set(rule['antecedent'])} → {set(rule['consequent'])}")
    print(f"   Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")



import seaborn as sns
# Flatten transaction list
from collections import Counter
item_counts = Counter([item for transaction in transaction_list for item in transaction])
# Get top 10 frequent items
top_items = item_counts.most_common(10)
items, counts = zip(*top_items)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(items), y=list(counts), palette='viridis')
plt.title("Top 10 Frequent Items")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()



if top_rules:
    strong_rule = top_rules[0]
    antecedent = set(strong_rule['antecedent'])
    consequent = set(strong_rule['consequent'])
    print(f"\n📌 Strong Rule: {antecedent} → {consequent}")
    print(f"Support: {strong_rule['support']:.2f}")
    print(f"Confidence: {strong_rule['confidence']:.2f}")
    print(f"Lift: {strong_rule['lift']:.2f}")

    print(f"\nInterpretation:\nIf a customer buys {antecedent}, they are highly likely to also buy {consequent}.")
    print(f"The lift value of {strong_rule['lift']:.2f} indicates this rule is {strong_rule['lift']:.2f} times more likely than random chance.")
else:
    print("⚠️ No association rules found. Try lowering min_support or min_confidence.")
