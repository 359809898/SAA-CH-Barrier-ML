# SAA-CH-Barrier-ML
Machine learning prediction and error diagnosis of C-H dissociation barriers on single-atom alloy surfaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("Database.csv")
# 打印每一列的列号和列名，方便核对
print('\n列位置（第几列 / 列名）:')
for i, col in enumerate(df.columns, 1):
    print(i, col)

#切特征
label = 'Barrier'

drop_names = [
    # 结果列必须全部删掉
    "Barrier", "rate_723.15", "rate_area_723",
    # 基础信息列（不做特征）
    "Host", "Surface", "area_fraction", "Doped",
    "host_symmetry_symbol",
    # 垃圾列
    "Unnamed: 97", "Unnamed: 98", "Unnamed: 99",
    # 分类标签列（不做回归特征）
    "site1", "site2", "site3", "site4",
]

#特征
feature_cols = [c for c in df.columns if c not in drop_names and c != label]
print('\n特征的数量:' ,len(feature_cols))
X = df[feature_cols]
y = df[label]
print(X.shape)
print(y.shape)

# 只保留数值型特征，防止混入文字列
X = X.select_dtypes(include=[np.number])
print('只保留数值特征后 X 形状:', X.shape)

# 缺失值用平均值填一下
X = X.fillna(X.mean())

# 标签有缺失的样本去掉
mask = y.notna()
X = X[mask]
y = y[mask]

print('最终 X 形状:', X.shape)
print('最终 y 形状:', y.shape)
"""
特征的数量: 84
(10950, 84)
(10950,)
"""
#划分数据集
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train.isnull().sum())
print(X_test.isnull().sum())
print(X_train.shape)
print(X_test.shape) 

#线性回归模型
Lr_model = LinearRegression()
Lr_model.fit(X_train, y_train)
y_predict = Lr_model.predict(X_test)
print('线性回归 预测值(前5个):', y_predict[:5])
print('线性回归 真实值(前5个):', y_test.values[:5])

#随机森林模型
Rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
Rf_model.fit(X_train, y_train)
y_predict_rf = Rf_model.predict(X_test)
print('随机森林 预测值(前5个):', y_predict_rf[:5])
print('随机森林 真实值(前5个):', y_test.values[:5])

#梯度提升模型
Gb_model = GradientBoostingRegressor(random_state=42)
Gb_model.fit(X_train, y_train)
y_predict_gb = Gb_model.predict(X_test)
print('梯度提升 预测值(前5个):', y_predict_gb[:5])
print('梯度提升 真实值(前5个):', y_test.values[:5])

# ========== 统一计算 R² 和 RMSE ==========
print('\n===== 线性回归 =====')
MSE = mean_squared_error(y_test, y_predict)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_predict)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('R²:', R2)

print('\n===== 随机森林 =====')
MSE = mean_squared_error(y_test, y_predict_rf)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_predict_rf)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('R²:', R2)

print('\n===== 梯度提升 =====')
MSE = mean_squared_error(y_test, y_predict_gb)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_predict_gb)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('R²:', R2)
"""
MSE: 0.0337691195470995
RMSE: 0.18376376015716347
R²: 0.7292706511052611

===== 随机森林 =====
MSE: 0.0025619670727506693
RMSE: 0.050615877674408345
R²: 0.9794605341567123

===== 梯度提升 =====
MSE: 0.007954475537407996
RMSE: 0.08918786653692304
R²: 0.936228423721916
前15个重要特征:
                          feature  importance
46  doped_weighted_surface_energy    0.447989
73                 com_top_d-band    0.107626
1      host_surface_work_function    0.052616
28              host_molar_volume    0.051162
74                 com_all_d-band    0.038819
17                host_d_e_number    0.034477
0             host_surface_energy    0.023467
68        doped_electronegativity    0.020505
78     com_cha2_electronegativity    0.019316
26              host_mendeleev_no    0.017443
77      com_cha_electronegativity    0.014137
76      com_top_electronegativity    0.010821
2             host_surface_efermi    0.010728
6                   CN-B3+1-top05    0.009310
80           com_top_out_e_number    0.008638
"""
importances = Rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
})
# 按重要性从大到小排序
feature_importance = feature_importance.sort_values(
    by='importance', ascending=False
)
print('前15个重要特征:')
print(feature_importance.head(15))

#########################################################################################################################

#GroupKFold泄漏检查
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np

# 构造体系 ID：同一 SAA 体系的不同 C-H 步骤不能分到训练/测试两边
df_clean = df[df['Barrier'].notna()].copy()
df_clean['system_id'] = (
    df_clean['Host'].astype(str) + '_' +
    df_clean['Surface'].astype(str) + '_' +
    df_clean['Doped'].astype(str)
)
X_all = df_clean[feature_cols].select_dtypes(include=[np.number]).fillna(0)
y_all = df_clean['Barrier']
groups = df_clean['system_id']

gkf = GroupKFold(n_splits=5)
scores_r2 = cross_val_score(
    RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    X_all, y_all, groups=groups, cv=gkf, scoring='r2'
)
scores_rmse = -cross_val_score(
    RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    X_all, y_all, groups=groups, cv=gkf, scoring='neg_root_mean_squared_error'
)
print('GroupKFold R2:', scores_r2.mean(), '±', scores_r2.std())
print('GroupKFold RMSE:', scores_rmse.mean(), '±', scores_rmse.std())
"""
GroupKFold R2: 0.7992140695666882 ± 0.15698999516734058
GroupKFold RMSE: 0.141223188765924 ± 0.0591292926396604
按照0.3划分
RMSE: 0.050615877674408345
R²: 0.9794605341567123
"""

# 重新跑一次 GroupKFold，拿 out-of-fold 预测
gkf = GroupKFold(n_splits=5)
oof_pred = np.full(len(X_all), np.nan)
fold_id = np.full(len(X_all), -1)

for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
    m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    m.fit(X_all.iloc[tr], y_all.iloc[tr])
    oof_pred[te] = m.predict(X_all.iloc[te])
    fold_id[te] = fold

ana = df_clean.copy()
ana['pred'] = oof_pred
ana['residual'] = ana['pred'] - ana['Barrier']
ana['abs_err'] = ana['residual'].abs()
ana['fold'] = fold_id

# 逐折
print('=== 逐折 ===')
for f in sorted(ana['fold'].unique()):
    sub = ana[ana['fold'] == f]
    print(f"Fold {f}: n={len(sub)}, R2={r2_score(sub['Barrier'], sub['pred']):.3f}, "
          f"RMSE={np.sqrt(((sub['pred']-sub['Barrier'])**2).mean()):.3f}, "
          f"Host top: {sub['Host'].value_counts().head(3).to_dict()}")

# 按 Host
print('\n=== 按 Host MAE ===')
host_err = ana.groupby('Host').agg(
    n=('abs_err','size'),
    MAE=('abs_err','mean'),
    bias=('residual','mean')
).sort_values('MAE', ascending=False)
print(host_err.head(10))

# 按 Doped
print('\n=== 按 Doped MAE ===')
doped_err = ana.groupby('Doped').agg(
    n=('abs_err','size'),
    MAE=('abs_err','mean'),
    bias=('residual','mean')
).sort_values('MAE', ascending=False)
print(doped_err.head(10))
"""
=== 按 Host MAE ===
        n       MAE      bias
Host                         
Cu    390  0.157651  0.031631
Ru    360  0.128230  0.007868
Ni    390  0.125470  0.014218
Cr    390  0.121754  0.024771
Pt    390  0.118277 -0.015232
Fe    390  0.117362  0.031046
Au    390  0.116572 -0.013791
Ag    390  0.116443  0.011543
Rh    390  0.115364  0.010158
Re    360  0.112869  0.000651

=== 按 Doped MAE ===
         n       MAE      bias
Doped                         
Cu     365  0.413071 -0.412866  表示Cu里面有365条数据 误差最大
Pd     365  0.403040  0.403014
Ag     365  0.160040 -0.158628
Au     365  0.144790  0.143414
Re     365  0.099512  0.093702
Mo     365  0.098870 -0.089234
W      365  0.085022 -0.009395
Co     365  0.084404  0.083330
Ni     365  0.080956 -0.057226
Ir     365  0.076034  0.024636
"""

#先画一下图然后分析一下Cu和Pd误差为什么这么大
plt.figure(figsize=(5,5))
plt.scatter(ana['Barrier'], ana['pred'], s=6, alpha=0.4, c='#4C72B0')
lim = [ana['Barrier'].min(), ana['Barrier'].max()]
plt.plot(lim, lim, 'r--')
plt.xlabel('DFT barrier (eV)')
plt.ylabel('OOF predicted barrier (eV)')
plt.title(f"GroupKFold: R²={r2_score(ana['Barrier'], ana['pred']):.3f}")
plt.tight_layout()
plt.show()


methods = ['Random 70/30', 'GroupKFold']
r2_vals = [0.979, 0.799]
rmse_vals = [0.051, 0.141]
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].bar(methods, r2_vals, color=['#999', '#4C72B0'])
axes[0].set_ylabel('R²')
axes[0].set_title('R²: Random vs GroupKFold')
axes[1].bar(methods, rmse_vals, color=['#999', "#B0974C"])
axes[1].set_ylabel('RMSE (eV)')
axes[1].set_title('RMSE: Random vs GroupKFold')
plt.tight_layout()
plt.show()
