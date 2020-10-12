import pandas as pd
import matplotlib.pyplot as plt

### TSP ####
df = pd.read_csv("TSP_ALL.csv")
df2 = df.groupby(["algorithm", "iterations"]).agg({"fitness":"mean", "time":"mean"}).reset_index()

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Fitness Score")
ax.set_title("Travelling Salesman Problem: Fitness Score")
for i in df2["algorithm"].unique():
	temp_df = df2[df2["algorithm"]==i]
	temp_val = list(temp_df["fitness"])
	temp_iteration = list(temp_df["iterations"])
	ax.plot(temp_iteration, temp_val, marker='o', label=i)
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Time in Seconds (Log)")
ax.set_yscale('log')
ax.set_title("Travelling Salesman Problem: Timing")
for i in df2["algorithm"].unique():
	temp_df = df2[df2["algorithm"]==i]
	temp_val = list(temp_df["time"])
	temp_iteration = list(temp_df["iterations"])
	ax.plot(temp_iteration, temp_val, marker='o', label=i)
ax.legend()
plt.show()


### CP ####

df = pd.read_csv("CP_ALL.csv")
df2 = df.groupby(["algorithm", "iterations"]).agg({"fitness":"mean", "time":"mean"}).reset_index()

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Fitness Score")
ax.set_title("Continous Peaks Problem: Fitness Score")
for i in df2["algorithm"].unique():
	temp_df = df2[df2["algorithm"]==i]
	temp_val = list(temp_df["fitness"])
	temp_iteration = list(temp_df["iterations"])
	ax.plot(temp_iteration, temp_val, marker='o', label=i)
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Time in Seconds (Log)")
ax.set_yscale('log')
ax.set_title("Continous Peaks Problem: Timing")
for i in df2["algorithm"].unique():
	temp_df = df2[df2["algorithm"]==i]
	temp_val = list(temp_df["time"])
	temp_iteration = list(temp_df["iterations"])
	ax.plot(temp_iteration, temp_val, marker='o', label=i)
ax.legend()
plt.show()


### KP ####

df = pd.read_csv("KP_ALL.csv")
df2 = df.groupby(["algorithm", "iterations"]).agg({"fitness":"mean", "time":"mean"}).reset_index()

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Fitness Score")
ax.set_title("Knapsack Problem: Fitness Score")
for i in df2["algorithm"].unique():
	temp_df = df2[df2["algorithm"]==i]
	temp_val = list(temp_df["fitness"])
	temp_iteration = list(temp_df["iterations"])
	ax.plot(temp_iteration, temp_val, marker='o', label=i)
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Time in Seconds (Log)")
ax.set_yscale('log')
ax.set_title("Knapsack Problem: Timing")
for i in df2["algorithm"].unique():
	temp_df = df2[df2["algorithm"]==i]
	temp_val = list(temp_df["time"])
	temp_iteration = list(temp_df["iterations"])
	ax.plot(temp_iteration, temp_val, marker='o', label=i)
ax.legend()
plt.show()


