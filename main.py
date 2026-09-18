import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

file_path = "telemetry_dzz_sem1.csv"
df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("--- 1. Первые 5 строк таблицы ---")
print(df.head())

print("\n--- 2. Размер набора данных ---")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

print("\n--- 3. Основные статистические характеристики ---")
print(df.describe())

df["rule_temp"] = df["temperature"] > 50.0
df["rule_volt"] = df["voltage"] < 24.0
df["rule_curr"] = df["current"] > 8.0
df["rule_omega"] = df["angular_velocity"] > 1.0

df["anomaly"] = (
    df["rule_temp"] | df["rule_volt"] | df["rule_curr"] | df["rule_omega"]
).astype(int)

print("\n=== 1. АНАЛИЗ ПАРАМЕТРОВ ПО РЕЖИМАМ ===")
mode_stats = df.groupby("mode")[
    ["temperature", "voltage", "current", "angular_velocity"]
].agg(["mean", "std", "min", "max"])
print(mode_stats.round(2))

mean_temp = df["temperature"].mean()
std_temp = df["temperature"].std()

df["anom_1sigma"] = (
    np.abs(df["temperature"] - mean_temp) > 1 * std_temp
).astype(int)
df["anom_2sigma"] = (
    np.abs(df["temperature"] - mean_temp) > 2 * std_temp
).astype(int)
df["anom_3sigma"] = (
    np.abs(df["temperature"] - mean_temp) > 3 * std_temp
).astype(int)

print("\n=== 2. СТАТИСТИЧЕСКИЙ ДЕТЕКТОР (Температура) ===")
print(f"Среднее: {mean_temp:.2f}°C, Std: {std_temp:.2f}°C")
print(
    f"Аномалий по 1σ (> {mean_temp + std_temp:.2f}°C): {df['anom_1sigma'].sum()} точек"
)
print(
    f"Аномалий по 2σ (> {mean_temp + 2*std_temp:.2f}°C): {df['anom_2sigma'].sum()} точек"
)
print(
    f"Аномалий по 3σ (> {mean_temp + 3*std_temp:.2f}°C): {df['anom_3sigma'].sum()} точек"
)

for col in ["temperature", "voltage", "current", "angular_velocity"]:
    df[f"{col}_diff"] = df[col].diff().abs()

df["det1"] = (
    (df["temperature"] > 50.0)
    | (df["voltage"] < 24.0)
    | (df["current"] > 8.0)
    | (df["angular_velocity"] > 1.0)
).astype(int)

df["det2"] = (
    (
        np.abs(df["temperature"] - df["temperature"].mean())
        > 2 * df["temperature"].std()
    )
    | (
        np.abs(df["voltage"] - df["voltage"].mean())
        > 2 * df["voltage"].std()
    )
    | (
        np.abs(df["current"] - df["current"].mean())
        > 2 * df["current"].std()
    )
    | (
        np.abs(df["angular_velocity"] - df["angular_velocity"].mean())
        > 2 * df["angular_velocity"].std()
    )
).astype(int)

score = (
    (df["temperature"] > 40.0).astype(int)
    + (df["voltage"] < 26.0).astype(int)
    + (df["current"] > 6.0).astype(int)
    + (df["angular_velocity"] > 0.5).astype(int)
    + (df["temperature_diff"] > 3.0).astype(int)
    + (df["voltage_diff"] > 1.0).astype(int)
)
df["det3"] = (score >= 2).astype(int)

print("\n=== 5. СРАВНЕНИЕ ТРЕХ ДЕТЕКТОРОВ ===")
print(f"Детектор №1 (Жёсткие пороги):       {df['det1'].sum()} точек")
print(f"Детектор №2 (Статистический 2-σ):   {df['det2'].sum()} точек")
print(f"Детектор №3 (Комбинация баллов):    {df['det3'].sum()} точек")

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Детектор 1
axes[0].plot(df["timestamp"], df["temperature"], color="gray", alpha=0.5)
anom1 = df[df["det1"] == 1]
axes[0].scatter(
    anom1["timestamp"],
    anom1["temperature"],
    color="red",
    s=20,
    label="Детектор №1 (Пороги)",
)
axes[0].set_title(
    f"Детектор №1 — Пороги из Задания №1 ({df['det1'].sum()} точек)"
)
axes[0].set_ylabel("Temp (°C)")
axes[0].legend(loc="upper right")

# Детектор 2
axes[1].plot(df["timestamp"], df["temperature"], color="gray", alpha=0.5)
anom2 = df[df["det2"] == 1]
axes[1].scatter(
    anom2["timestamp"],
    anom2["temperature"],
    color="orange",
    s=20,
    label="Детектор №2 (Статистика 2σ)",
)
axes[1].set_title(
    f"Детектор №2 — Статистический ({df['det2'].sum()} точек)"
)
axes[1].set_ylabel("Temp (°C)")
axes[1].legend(loc="upper right")

# Детектор 3
axes[2].plot(df["timestamp"], df["temperature"], color="gray", alpha=0.5)
anom3 = df[df["det3"] == 1]
axes[2].scatter(
    anom3["timestamp"],
    anom3["temperature"],
    color="green",
    s=20,
    label="Детектор №3 (Комбинация)",
)
axes[2].set_title(
    f"Детектор №3 — Комбинация признаков и diff ({df['det3'].sum()} точек)"
)
axes[2].set_ylabel("Temp (°C)")
axes[2].set_xlabel("Время")
axes[2].legend(loc="upper right")

plt.suptitle(
    "Сравнение обнаружения аномалий тремя детекторами", fontsize=14
)
plt.tight_layout()
plt.show()