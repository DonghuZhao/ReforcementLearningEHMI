from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

# 创建贝叶斯网络模型
bn = BayesianNetwork([('I_0', 'Dp_0'), ('I_0', 'Dv_0'), ('I_0', 'D_0'), ('Dp_0', 'V_0'), ('Dv_0', 'V_0')])

# 定义节点的取值范围和间隔
Dp_0_values = list(range(-10, 61, 5))
Dv_0_values = list(range(-10, 11))
V_0_values = list(range(0, 21, 2))
D_0_values = list(range(0, 61, 5))
# 添加节点的条件概率分布
cpd_I_0 = TabularCPD(variable='I_0', variable_card=2, values=[[0.5], [0.5]])  # I_0 是一个二值随机变量

# Dp_0 和 Dv_0 的取值不依赖于 I_0，所以每个可能取值的概率应该相同
cpd_Dp_0 = TabularCPD(variable='Dp_0', variable_card=len(Dp_0_values),
                      values=[[1/len(Dp_0_values)] * 2 for _ in range(len(Dp_0_values))],
                      evidence=['I_0'], evidence_card=[2])

cpd_Dv_0 = TabularCPD(variable='Dv_0', variable_card=len(Dv_0_values),
                      values=[[1/len(Dv_0_values)] * 2 for _ in range(len(Dv_0_values))],
                      evidence=['I_0'], evidence_card=[2])

cpd_D_0 = TabularCPD(variable='D_0', variable_card=len(D_0_values),
                      values=[[1/len(D_0_values)] * 2 for _ in range(len(D_0_values))],
                      evidence=['I_0'], evidence_card=[2])

# V_0 的取值取决于其父节点 Dp_0 和 Dv_0
# 这里我们为每一对 (Dp_0, Dv_0) 的组合分配随机的概率值
cpd_V_0 = TabularCPD(variable='V_0', variable_card=len(V_0_values),
                     values=[[1 / len(V_0_values)] * len(Dp_0_values) * len(Dv_0_values) for _ in range(len(V_0_values))],
                     evidence=['Dp_0', 'Dv_0'], evidence_card=[len(Dp_0_values), len(Dv_0_values)])


# 添加边的条件概率分布到模型中
bn.add_cpds(cpd_I_0, cpd_Dp_0, cpd_Dv_0, cpd_D_0, cpd_V_0)

# 检查模型的结构和参数是否正确
print(bn.check_model())

print("-------------inferring--------------")

evidence_X = {"D_0" :(0, 1)}

# 进行推断
inference = VariableElimination(bn)
posterior_I = inference.query(variables=['I_0'], evidence=evidence_X)

# 打印推断结果
print("后验概率分布 P(I|X):")
print(posterior_I)