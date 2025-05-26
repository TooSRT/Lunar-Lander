import pandas as pd
import matplotlib.pyplot as plt


df_64 = pd.read_csv("best_scores_hdim=64.csv")
df_64_2 = pd.read_csv("best_scores_hdim=64_2.csv")
df_64_3 = pd.read_csv("best_scores_hdim=64_3.csv")
df_32 = pd.read_csv("best_scores_hdim=32.csv")
df_16 = pd.read_csv("best_scores_hdim=16.csv")
df_8 = pd.read_csv("best_scores_hdim=8.csv")
df_8_1 = pd.read_csv("best_scores_hdim=8_1.csv")
df_8_2 = pd.read_csv("best_scores_hdim=8_2.csv")
df_8_3 = pd.read_csv("best_scores_hdim=8_3.csv")

plt.figure(figsize=(8, 5))

plt.plot(df_64["Iteration"], df_64["Best_Fitness"], marker='o', linestyle='-',  label="hdim=64")
plt.plot(df_32["Iteration"], df_32["Best_Fitness"], marker='o', linestyle='-',  label="hdim=32")
plt.plot(df_16["Iteration"], df_16["Best_Fitness"], marker='o', linestyle='-',  label="hdim=16")
plt.plot(df_8["Iteration"], df_8["Best_Fitness"], marker='o', linestyle='-', label="hdim=8")

plt.axhline(y=200, color='black', linestyle='--', linewidth=1, label="Target Score (200)")

plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Best Fitness Evolution depending on the number of neurons hdim")
plt.legend()
plt.grid(True)

plt.savefig("best_fitness_evolution_ar.png")
plt.show()

plt.plot(df_8_1["Iteration"], df_8_1["Best_Fitness"], marker='.', linestyle='-', label="hdim=8 (1)")
plt.plot(df_8_2["Iteration"], df_8_2["Best_Fitness"], marker='.', linestyle='-', label="hdim=8 (2)")
plt.plot(df_8_3["Iteration"], df_8_3["Best_Fitness"], marker='.', linestyle='-', label="hdim=8 (3)")
plt.plot(df_64["Iteration"], df_64["Best_Fitness"], marker='^', linestyle='-',  label="hdim=64 (1)")
plt.plot(df_64_2["Iteration"], df_64_2["Best_Fitness"], marker='^', linestyle='-',  label="hdim=64 (2)")
plt.plot(df_64_3["Iteration"], df_64_3["Best_Fitness"], marker='^', linestyle='-',  label="hdim=64 (3)")

plt.axhline(y=200, color='black', linestyle='--', linewidth=1, label="Target Score (200)")

plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Best Fitness Evolution depending on the number of neurons hdim")
plt.legend()
plt.grid(True)

plt.savefig("best_fitness_evolution_ar2.png")
plt.show()

