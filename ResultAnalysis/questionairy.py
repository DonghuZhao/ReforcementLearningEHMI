import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和英文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']

file_path = r"C:\Users\11917\OneDrive - tongji.edu.cn\硕士\毕业论文\结题\实验\300042154_按序号_自动驾驶交互实验场景评估问卷_17_17.xlsx"

STRATEGY = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'A',
    5: 'B',
    6: 'C',
    7: 'A',
    8: 'B',
    9: 'C',
}

SCENARIO = {
    1: 'L',
    2: 'L',
    3: 'L',
    4: 'SF',
    5: 'SF',
    6: 'SF',
    7: 'S',
    8: 'S',
    9: 'S',
}


intention = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

definition = copy.deepcopy(intention)
timing = copy.deepcopy(intention)
safety = copy.deepcopy(intention)
helpful = copy.deepcopy(intention)

data = pd.read_excel(file_path, sheet_name="Sheet1", header=0)
print(data.columns)
for i in range(1, 10):
    intention[i] = list(data[f"{5 * (i - 1) + 2}、您认为对向左转车的意图是"])
    definition[i] = list(data[f"{5 * (i - 1) + 3}、您认为对向左转车的意图（先行\让行）是否清晰"])
    timing[i] = list(data[f"{5 * (i - 1) + 4}、你在交互过程的哪一阶段能够完全明确左转车的意图？"])
    safety[i] = list(data[f"{5 * (i - 1) + 5}、您对此次交互过程的安全评价"])
    helpful[i] = list(data[f"{5 * (i - 1) + 6}、对向车显示的EHMI信息对你是否有帮助"])
print("intention:\n", intention)
print("definition:\n", definition)
print("timing:\n", timing)
print("safety:\n", safety)
print("helpful:\n", helpful)

for i in range(1, 10):
    print(f"Scenario_{i}_average_definition: {sum(intention[i]) / len(intention[i])}")
    print(f"Scenario_{i}_average_timing: {sum(timing[i]) / len(timing[i])}")
    print(f"Scenario_{i}_average_safety: {sum(safety[i]) / len(safety[i])}")
    print(f"Scenario_{i}_average_helpful: {sum(helpful[i]) / len(helpful[i])}")

# 创建一个 ExcelWriter 对象
with pd.ExcelWriter("result.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    # 将每个 DataFrame 写入不同的工作表
    pd.DataFrame(definition).to_excel(writer, sheet_name="definition", index=False)
    pd.DataFrame(timing).to_excel(writer, sheet_name="timing", index=False)
    pd.DataFrame(safety).to_excel(writer, sheet_name="safety", index=False)
    pd.DataFrame(helpful).to_excel(writer, sheet_name="helpful", index=False)


# definition 绘制箱型图
# 将数据转换为 Pandas DataFrame
df_definition = []
for key, values in definition.items():
    for value in values:
        df_definition.append({'Strategy': STRATEGY[key], 'Scenario': SCENARIO[key], 'Value': value})
df_definition = pd.DataFrame(df_definition)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_definition, x='Scenario', y='Value', hue='Strategy', palette='PuBu')
plt.title('AV车意图清晰度评价', fontsize=20)
plt.xlabel('Scenario', fontsize=20)
plt.ylabel('definition', fontsize=20)
plt.xticks(ticks=[0,1,2], labels=['AV左转', 'AV直行优势', 'AV直行劣势'], fontsize=15)
plt.yticks(fontsize=12)
plt.yticks(np.arange(0, 6, 1))
plt.legend(title='Strategy', fontsize=15)
plt.show()

# timing 绘制箱型图
# 将数据转换为 Pandas DataFrame
df_timing = []
for key, values in timing.items():
    for value in values:
        df_timing.append({'Strategy': STRATEGY[key], 'Scenario': SCENARIO[key], 'Value': value})
df_timing = pd.DataFrame(df_timing)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_timing, x='Scenario', y='Value', hue='Strategy', palette='PuBu')
plt.title('AV车意图识别早晚评价', fontsize=20)
plt.xlabel('Scenario', fontsize=20)
plt.ylabel('timing', fontsize=20)
plt.xticks(ticks=[0,1,2], labels=['AV左转', 'AV直行优势', 'AV直行劣势'], fontsize=15)
plt.yticks(fontsize=12)
plt.yticks(np.arange(0, 6, 1))
plt.legend(title='Strategy', fontsize=15)
plt.show()

# safety 绘制箱型图
# 将数据转换为 Pandas DataFrame
df_safety = []
for key, values in safety.items():
    for value in values:
        df_safety.append({'Strategy': STRATEGY[key], 'Scenario': SCENARIO[key], 'Value': value})
df_safety = pd.DataFrame(df_safety)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_safety, x='Scenario', y='Value', hue='Strategy', palette='PuBu')
plt.title('交互安全性评价', fontsize=20)
plt.xlabel('Scenario', fontsize=20)
plt.ylabel('safety', fontsize=20)
plt.xticks(ticks=[0,1,2], labels=['AV左转', 'AV直行优势', 'AV直行劣势'], fontsize=15)
plt.yticks(fontsize=12)
plt.yticks(np.arange(0, 6, 1))
plt.legend(title='Strategy', fontsize=15)
plt.show()

# helpful 绘制箱型图
# 将数据转换为 Pandas DataFrame
df_helpful = []
for key, values in helpful.items():
    if key in [1, 4, 7]:
        continue
    for value in values:
        df_helpful.append({'Strategy': STRATEGY[key], 'Scenario': SCENARIO[key], 'Value': value})
df_helpful = pd.DataFrame(df_helpful)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_helpful, x='Scenario', y='Value', hue='Strategy', palette='PuBu')
plt.title('EHMI有用性评价', fontsize=20)
plt.xlabel('Scenario', fontsize=20)
plt.ylabel('helpful', fontsize=20)
plt.xticks(ticks=[0,1,2], labels=['AV左转', 'AV直行优势', 'AV直行劣势'], fontsize=15)
plt.yticks(fontsize=12)
plt.yticks(np.arange(0, 6, 1))
plt.legend(title='Strategy', fontsize=15)
plt.show()