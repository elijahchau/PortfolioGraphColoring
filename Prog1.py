# %%
%pip install yfinance pandas numpy matplotlib networkx

# %%
import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

stocks = [
    # AI hyperscalers and software
    "GOOGL", "GOOG", "MSFT", "AMZN", "META", "ORCL", "CRM",
    "PLTR", "SNOW", "NOW", "IBM", "ADBE",

    # AI chips and semiconductor design
    "NVDA", "AMD", "AVGO", "MRVL", "ARM", "QCOM", "INTC",
    "TSM", "ASML", "MU",

    # Memory and storage
    "MU", "SNDK", "SKHY",

    # AI data centers and compute infrastructure
    "NBIS", "IREN", "CIFR", "CRWV",

    # Other energy
    "XOM", "CVX", "COP", "SLB", "FSLR", "ENPH",

    # Financials
    "JPM", "BAC", "GS", "MS", "C", "V", "MA", "BRK-B",

    # Healthcare and pharmaceuticals
    "LLY", "JNJ", "ABBV", "MRK", "UNH", "ISRG", "PFE",

    # Aerospace
    "RKLB", "ASTS", "SPCX", "GE", "RTX", "LMT", "BA",

    # Gold and precious metals
    "NEM", "AEM", "GOLD",

    # Broad market benchmarks
    "SPY", "QQQ"
]

prices = yf.download(
    stocks,
    period="1y",
    auto_adjust=True,
    progress=False
)["Close"]

prices = prices.dropna()
returns = prices.pct_change().dropna()

returns.head()

# %%
corr = returns.corr()

plt.figure(figsize=(9, 7))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(stocks)), stocks, rotation=45)
plt.yticks(range(len(stocks)), stocks)
plt.title("Stock Correlation Matrix")
plt.tight_layout()
plt.show()

# %%
threshold = 0.6

G = nx.Graph()
G.add_nodes_from(stocks)

for i in stocks:
    for j in stocks:
        if i != j and corr.loc[i, j] >= threshold:
            G.add_edge(i, j, weight=corr.loc[i, j])

print("Number of stocks:", G.number_of_nodes())
print("Number of connections:", G.number_of_edges())

# %%
colors = nx.coloring.greedy_color(G, strategy="largest_first")
node_colors = [colors[node] for node in G.nodes()]
pos = nx.spring_layout(G, seed=42, k=0.8)

plt.figure(figsize=(16, 12))

nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    cmap=plt.cm.Accent, 
    node_size=1800
)

nx.draw_networkx_labels(
    G, pos,
    font_size=9,
    font_weight="bold"
)

nx.draw_networkx_edges(
    G, pos,
    edge_color="gray",
    width=1.5,
    alpha=0.7
)

edge_labels = nx.get_edge_attributes(G, "weight")
edge_labels = {edge: f"{weight:.2f}" for edge, weight in edge_labels.items()}

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=8,
    font_color="red",
    rotate=False
)

plt.title(f"Stock Correlation Graph (Threshold = {threshold})")
plt.axis("off")
plt.tight_layout()
plt.show()

print("Stock color assignments:")
for stock, color in sorted(colors.items()):
    print(f"{stock}: Color {color}")

# %%
pairs = []

for i in range(len(stocks)):
    for j in range(i + 1, len(stocks)):
        pairs.append({
            "Stock 1": stocks[i],
            "Stock 2": stocks[j],
            "Correlation": corr.loc[stocks[i], stocks[j]]
        })

results = pd.DataFrame(pairs)
results = results.sort_values("Correlation", ascending=False)

results.head(10)


