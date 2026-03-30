import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import corner  # 需要 pip install corner
from scipy.stats import binned_statistic
from scipy.interpolate import UnivariateSpline
import os
import csv
# based on env name of base
# ================= 配置 =================
# 建议将星团名提取为变量，方便保存文件
CLUSTER_NAME = "NGC5272"
FILE_PATH = 'golden_samples/HST_56GC/ngc5272/hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc5272_multi_v1_catalog-meth1.txt'

SAMPLE_SIZE = 5000 
RANDOM_SEED = 42

# NGC 0288 距离模数 (m-M)v ~ 14.8, TO点 ~ 18.0
# 选择主序带 (MS) 进行分析，通常选 TO 以下 1.5 - 4 mag
# 19.0 - 21.5 是一个非常合适的深主序区间.
#  运行 plt_cmd.py 结合三个CMD人工选择主序带 
MAG_MIN, MAG_MAX = 19, 21.5
# MAG_MIN, MAG_MAX = 17.5, 20 # for ngc 104  
 
# =======================================

def get_ridge_line(mag, color, bins=20):
    bin_edges = np.linspace(mag.min(), mag.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_medians, _, _ = binned_statistic(mag, color, statistic='median', bins=bin_edges)
    valid = ~np.isnan(bin_medians)
    if valid.sum() < 5: return None
    return UnivariateSpline(bin_centers[valid], bin_medians[valid], k=3, s=0)

def preprocess_data():
    if not os.path.exists(FILE_PATH): raise FileNotFoundError(f"找不到文件: {FILE_PATH}")
    print(f"⏳ [{CLUSTER_NAME}] 读取并预处理数据...")
    df = pd.read_csv(FILE_PATH, sep='\s+', comment='#', header=None)
    cols = {2:'F275W', 4:'Q275', 8:'F336W', 10:'Q336', 14:'F438W', 16:'Q438',
            20:'F606W', 22:'Q606', 26:'F814W', 28:'Q814', 32:'Prob'}
    if df.shape[1] < 33: return None
    df = df.rename(columns=cols)
    
    # 筛选
    mask = (df['F606W'] >= MAG_MIN) & (df['F606W'] <= MAG_MAX) & (df['Prob'] > 90) & \
           (df['Q275']>0.9) & (df['Q606']>0.9) & (df['Q814']>0.9)
    data = df[mask].copy()
    
    data['C_pseudo'] = (data['F275W'] - data['F336W']) - (data['F336W'] - data['F438W'])
    data['Color_Opt'] = data['F606W'] - data['F814W']
    
    # 垂直化
    sp_pseudo = get_ridge_line(data['F606W'], data['C_pseudo'])
    sp_opt = get_ridge_line(data['F606W'], data['Color_Opt'])
    
    if sp_pseudo is None or sp_opt is None: return None

    # 校正主序弯曲
    data['Delta_Pseudo'] = data['C_pseudo'] - sp_pseudo(data['F606W'])
    data['Delta_Opt'] = data['Color_Opt'] - sp_opt(data['F606W'])
    
    # 随机采样以加速MCMC
    if len(data) > SAMPLE_SIZE:
        print(f"   数据量 {len(data)} -> 采样至 {SAMPLE_SIZE}")
        data = data.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
        
    return data

def run_bayesian_mixture(data):
    obs_pop = data['Delta_Pseudo'].values
    obs_bin = data['Delta_Opt'].values
    obs_data = np.vstack([obs_pop, obs_bin]).T
    
    print(f"🚀 [{CLUSTER_NAME}] 构建模型 (N={len(obs_pop)})...")
    
    with pm.Model() as model:
        # --- Level 1: 场星 (Field Stars) ---
        w_field = pm.Beta('w_field', alpha=1, beta=20) 
        w_cluster = pm.Deterministic('w_cluster', 1 - w_field)
        
        # 场星宽分布
        mu_field = pm.Normal('mu_field', 0, 0.5, shape=2) 
        sigma_field = pm.HalfNormal('sigma_field', sigma=0.5, shape=2)

        # --- Level 2: 星族比例 (Populations) ---
        w_2g_internal = pm.Beta('w_2g_int', 5, 3) 
        w_1g_internal = 1 - w_2g_internal
        
        # 种群位置：强制 mu_1g > mu_2g 或使用 delta 用于识别
        mu_2g = pm.Normal('mu_2g', mu=-0.05, sigma=0.05)
        delta_pop = pm.HalfNormal('delta_pop', sigma=0.1)
        mu_1g = pm.Deterministic('mu_1g', mu_2g + delta_pop)
        
        sigma_pop = pm.HalfNormal('sigma_pop', sigma=0.04, shape=2)

        # --- Level 3: 双星比例 (Binary Fractions) ---
        # 重要：这是论文的核心物理参数
        f_bin_1g = pm.Beta('f_bin_1g', 1.5, 8.5)
        f_bin_2g = pm.Beta('f_bin_2g', 1.5, 8.5)
        
        sigma_single = pm.HalfNormal('sigma_single', sigma=0.015)
        # 截断正态分布：双星总是比单星红 (Offset > 0)
        mu_bin_offset = pm.TruncatedNormal('mu_bin_offset', mu=0.06, sigma=0.05, lower=0.02)
        sigma_bin = pm.HalfNormal('sigma_bin', sigma=0.08)

        # --- 混合权重 ---
        weights = pm.math.stack([
            w_cluster * w_2g_internal * (1 - f_bin_2g), # 2G Single
            w_cluster * w_2g_internal * f_bin_2g,       # 2G Binary
            w_cluster * w_1g_internal * (1 - f_bin_1g), # 1G Single
            w_cluster * w_1g_internal * f_bin_1g,       # 1G Binary
            w_field                                     # Field Stars
        ])
        
        # --- 协方差矩阵构建 ---
        def make_cov(sx, sy):
            return pm.math.stack([pm.math.stack([sx**2, 0.0]), pm.math.stack([0.0, sy**2])])

        dist1 = pm.MvNormal.dist(mu=[mu_2g, 0.0], cov=make_cov(sigma_pop[0], sigma_single))
        dist2 = pm.MvNormal.dist(mu=[mu_2g, mu_bin_offset], cov=make_cov(sigma_pop[0], sigma_bin))
        dist3 = pm.MvNormal.dist(mu=[mu_1g, 0.0], cov=make_cov(sigma_pop[1], sigma_single))
        dist4 = pm.MvNormal.dist(mu=[mu_1g, mu_bin_offset], cov=make_cov(sigma_pop[1], sigma_bin))
        dist5 = pm.MvNormal.dist(mu=mu_field, cov=make_cov(sigma_field[0], sigma_field[1]))
        
        # --- 似然函数 ---
        pm.Mixture('obs', w=weights, comp_dists=[dist1, dist2, dist3, dist4, dist5], observed=obs_data)
        
        # --- 追踪比率 ---
        ratio = pm.Deterministic('ratio_2g_1g', f_bin_2g / f_bin_1g)
        
        print("🎲 开始 MCMC 采样...")
        trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.95, return_inferencedata=True)
        
        print("🔮 生成后验预测检查(PPC)...")
        pm.sample_posterior_predictive(trace, model=model, extend_inferencedata=True)
        
    return trace

def save_and_visualize(trace, data):
    post = trace.posterior
    
    # 提取关键数组
    f1 = post['f_bin_1g'].values.flatten()
    f2 = post['f_bin_2g'].values.flatten()
    r = post['ratio_2g_1g'].values.flatten()
    w_field = post['w_field'].values.flatten()
    
    # 计算 HDI (95%)
    hdi_r = az.hdi(r, hdi_prob=0.95)
    
    # ---------------------------------------------------------
    # 1. 保存数值结果到 CSV (为 Summary Plot 做准备)
    # ---------------------------------------------------------
    results = {
        'Cluster': CLUSTER_NAME,
        'Mag_Range': f"{MAG_MIN}-{MAG_MAX}",
        'N_Sample': len(data),
        'f_bin_1g_mean': np.mean(f1),
        'f_bin_1g_std': np.std(f1),
        'f_bin_2g_mean': np.mean(f2),
        'f_bin_2g_std': np.std(f2),
        'ratio_mean': np.mean(r),
        'ratio_hdi_low': hdi_r[0],
        'ratio_hdi_high': hdi_r[1],
        'w_field_mean': np.mean(w_field)
    }
    
    csv_file = 'GC_Binary_Results_Summary.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    
    print(f"\n✅ 结果已追加保存至: {csv_file}")
    print(f"   -> Ratio: {results['ratio_mean']:.3f} [{hdi_r[0]:.3f}, {hdi_r[1]:.3f}]")
    
    # ---------------------------------------------------------
    # 2. 绘制 Corner Plot (Nature 级图表)
    # ---------------------------------------------------------
    print("📊 绘制 Corner Plot...")
    # 提取我们最关心的参数
    # 如果参数在 posterior 中是多维的，az.extract 会自动展平
    dataset = az.extract(trace, var_names=['f_bin_1g', 'f_bin_2g', 'ratio_2g_1g', 'w_field'])
    samples = dataset.to_dataframe()
    
    # 定义更漂亮的标签
    labels = [
        r"$f_{bin, 1G}$", 
        r"$f_{bin, 2G}$", 
        r"$Ratio (2G/1G)$", 
        r"$w_{field}$"
    ]
    
    # 使用 corner 库绘图
    fig = corner.corner(
        samples, 
        labels=labels,
        quantiles=[0.16, 0.5, 0.84], # 显示 1-sigma 范围
        show_titles=True, 
        title_fmt=".3f",
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        color="#0066CC",
        plot_density=True,
        plot_contours=True,
    )
    fig.suptitle(f"{CLUSTER_NAME} Posterior Distributions", fontsize=16)
    plt.savefig(f"{CLUSTER_NAME}_corner.png", dpi=300, bbox_inches='tight')
    plt.show() # 在 Notebook 中显示
    
    # ---------------------------------------------------------
    # 3. 绘制 PPC 和成分分离图 (常规验证)
    # ---------------------------------------------------------
    # 提取 PPC 数据
    ppc_data = trace.posterior_predictive['obs'].values.reshape(-1, 2)
    # 随机采样用于绘图
    idx = np.random.choice(len(ppc_data), size=min(5000, len(data)), replace=False)
    sim_obs = ppc_data[idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (A) 原始数据
    axes[0].scatter(data['Delta_Pseudo'], data['Delta_Opt'], s=2, alpha=0.3, c='k')
    axes[0].set_title(f"Observed Data ({CLUSTER_NAME})")
    axes[0].set_xlabel(r'$\Delta_{Pseudo}$'); axes[0].set_ylabel(r'$\Delta_{Opt}$')
    
    # (B) 模型数据
    axes[1].scatter(sim_obs[:, 0], sim_obs[:, 1], s=2, alpha=0.3, c='green')
    axes[1].set_title("Posterior Predictive Simulation")
    axes[1].set_xlabel(r'$\Delta_{Pseudo}$')
    
    # (C) 边缘直方图
    bins = np.linspace(-0.4, 0.4, 40)
    axes[2].hist(data['Delta_Pseudo'], bins=bins, density=True, histtype='step', color='k', lw=2, label='Obs')
    axes[2].hist(sim_obs[:, 0], bins=bins, density=True, histtype='step', ls='--', color='green', lw=2, label='Model')
    axes[2].set_title("Marginal Check (Pseudo Color)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{CLUSTER_NAME}_ppc_check.png")
    plt.close() # 避免在这一步显示，节省屏幕空间，已保存文件

if __name__ == "__main__":
    data_sample = preprocess_data()
    if data_sample is not None:
        try:
            # 运行模型
            final_trace = run_bayesian_mixture(data_sample)
            # 结果处理、保存与绘图
            save_and_visualize(final_trace, data_sample)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ {CLUSTER_NAME} 处理失败: {e}")
