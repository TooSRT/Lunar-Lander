import pandas as pd
import matplotlib.pyplot as plt


df_min = pd.read_csv("best_scores_min.csv")
df_mean = pd.read_csv("best_scores_mean.csv")
df_max = pd.read_csv("best_scores_max.csv")

plt.figure(figsize=(8, 5))

plt.plot(df_min["Iteration"], df_min["Best_Fitness"], marker='o', linestyle='-', color='blue', label="Min Fitness")
plt.plot(df_mean["Iteration"], df_mean["Best_Fitness"], marker='o', linestyle='-', color='green', label="Mean Fitness")
plt.plot(df_max["Iteration"], df_max["Best_Fitness"], marker='o', linestyle='-', color='red', label="Max Fitness")

plt.axhline(y=200, color='black', linestyle='--', linewidth=1, label="Target Score (200)")

plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Best Fitness Evolution")
plt.legend()
plt.grid(True)

plt.savefig("best_fitness_evolution.png")
plt.show()


