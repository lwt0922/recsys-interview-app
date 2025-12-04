import os
# --- 1. 必须放在第一行: 解决 Anaconda 冲突 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn

# --- 2. 页面配置 ---
st.set_page_config(
    page_title="推荐系统原理深度解析",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CSS 样式修复 (重点修复: 顶部字被切掉的问题) ---
st.markdown("""
    <style>
    /* 增加顶部内边距，防止标题被遮挡 */
    .block-container {
        padding-top: 3.5rem; 
        padding-bottom: 2rem;
    }
    /* 美化指标卡片 */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    /* 调整暗色模式下的文字颜色兼容 */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #464b59;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- 侧边栏 ---
st.sidebar.title("🎓 推荐系统教学")
st.sidebar.markdown("从入门到面试")
module = st.sidebar.radio(
    "课程章节:",
    [
        "1. 协同过滤 (基础篇)",
        "2. 矩阵分解 (进阶篇)",
        "3. 神经协同过滤 (深度学习)",
        "4. 面试模拟 (实战)"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("数据源：内置 5x5 教学矩阵")

# --- 辅助函数 ---
def get_synthetic_data():
    data = {
        '电影_A': [5, 4, 0, 1, 0],
        '电影_B': [0, 5, 4, 0, 2],
        '电影_C': [4, 0, 0, 2, 1],
        '电影_D': [1, 1, 0, 5, 4],
        '电影_E': [0, 2, 5, 4, 0]
    }
    return pd.DataFrame(data, index=[f'用户_{i}' for i in range(1, 6)])

# ==========================================
# 模块 1: 协同过滤
# ==========================================
if module == "1. 协同过滤 (基础篇)":
    st.title("📌 模块 1: 基于用户的协同过滤")
    st.markdown("**核心思想：** 既然我们口味相似，你喜欢的我也大概率喜欢。")

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.subheader("1. 交互式评分数据")
        df = get_synthetic_data()
        edited_df = st.data_editor(df, key="rating_grid", use_container_width=True)
        
        # 实时计算指标
        sparsity = (edited_df == 0).sum().sum() / edited_df.size
        st.metric("数据稀疏度 (Sparsity)", f"{sparsity:.1%}", 
                 delta="警惕冷启动" if sparsity > 0.8 else "数据正常",
                 delta_color="inverse")

    with col2:
        st.subheader("2. 相似度热力图")
        # 计算皮尔逊相关系数
        corr = edited_df.replace(0, np.nan).T.corr().fillna(0)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 模块 2: 矩阵分解 (增加可视化功能)
# ==========================================
elif module == "2. 矩阵分解 (进阶篇)":
    st.title("📌 模块 2: 矩阵分解 (Matrix Factorization)")
    st.markdown("**核心思想：** 这里的每个点都代表一个“隐向量”。**距离越近，代表越匹配。**")

    # 参数设置区
    with st.expander("⚙️ 模型超参数设置 (点此展开)", expanded=True):
        c1, c2, c3 = st.columns(3)
        K = c1.slider("隐向量维度 (K)", 2, 4, 2)
        epochs = c2.slider("训练轮数", 20, 200, 100)
        lr = c3.number_input("学习率", 0.001, 0.1, 0.01)

    # 训练逻辑
    df = get_synthetic_data()
    R = df.values
    N, M = R.shape
    
    np.random.seed(42)
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    
    # 模拟训练过程
    loss_history = []
    progress_bar = st.progress(0)
    
    for epoch in range(epochs):
        # 简单的 SGD 更新
        mask = R > 0
        error = R - np.dot(P, Q.T)
        error[~mask] = 0  # 只计算观测到的评分
        
        # 更新 (简化版无正则化，便于演示)
        grad_P = -2 * np.dot(error, Q)
        grad_Q = -2 * np.dot(error.T, P)
        
        P -= lr * grad_P
        Q -= lr * grad_Q
        
        loss = np.sum(error ** 2)
        loss_history.append(loss)
        if epoch % 10 == 0:
            progress_bar.progress(epoch / epochs)
            
    progress_bar.empty() # 清除进度条

    # --- 新增功能：可视化隐空间 ---
    col_viz, col_data = st.columns([1, 1])
    
    with col_viz:
        st.subheader("🌌 隐向量空间可视化 (Latent Space)")
        # 准备绘图数据
        if K >= 2:
            # 将用户和物品放在同一个 DataFrame 中
            user_df = pd.DataFrame(P[:, :2], columns=['x', 'y'])
            user_df['name'] = df.index
            user_df['type'] = '用户 (User)'
            
            item_df = pd.DataFrame(Q[:, :2], columns=['x', 'y'])
            item_df['name'] = df.columns
            item_df['type'] = '物品 (Item)'
            
            plot_df = pd.concat([user_df, item_df])
            
            fig = px.scatter(plot_df, x='x', y='y', color='type', text='name', 
                             title=f"用户与物品的二维映射 (K={K})",
                             symbol='type', size_max=15)
            fig.update_traces(textposition='top center')
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            st.info("👆 观察：如果某个用户离某部电影很近，说明模型预测他会喜欢这部电影。")
        else:
            st.warning("维度 K 必须 >= 2 才能进行二维可视化。")

    with col_data:
        st.subheader("📉 训练收敛曲线")
        st.line_chart(loss_history)
        
        # 展示预测结果对比
        st.subheader("🔍 预测评分 vs 真实评分")
        R_hat = np.dot(P, Q.T)
        comparison = pd.DataFrame({
            "真实值": R.flatten(),
            "预测值": R_hat.flatten()
        })
        # 只显示非0的真实评分
        st.dataframe(comparison[comparison["真实值"] > 0].head(5), use_container_width=True)

# ==========================================
# 模块 3: 神经协同过滤
# ==========================================
elif module == "3. 神经协同过滤 (深度学习)":
    st.title("📌 模块 3: 神经协同过滤 (NCF)")
    st.markdown("深度学习时代：不再只是点积，而是**非线性特征交叉**。")

    # 架构图 (使用 Graphviz 渲染)
    st.graphviz_chart("""
    digraph NCF {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor="#e1f5fe"];
        User [label="用户 ID"];
        Item [label="物品 ID"];
        
        node [fillcolor="#fff9c4"];
        Emb_U [label="用户 Embedding"];
        Emb_I [label="物品 Embedding"];
        
        node [fillcolor="#e0f2f1", shape=ellipse];
        Concat [label="拼接 (Concat)"];
        MLP [label="多层感知机 (MLP)"];
        Output [label="预测分数", shape=doublecircle, fillcolor="#ffccbc"];
        
        User -> Emb_U;
        Item -> Emb_I;
        Emb_U -> Concat;
        Emb_I -> Concat;
        Concat -> MLP;
        MLP -> Output;
    }
    """)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔮 Embedding 层探秘")
        user_id = st.selectbox("选择一个用户ID", range(5))
        emb_dim = st.slider("Embedding 维度", 4, 16, 8)
        
    with c2:
        st.markdown(f"**用户 {user_id} 的稠密向量表示：**")
        # 模拟 PyTorch Embedding
        vec = np.random.randn(emb_dim)
        st.code(str(np.round(vec, 3)), language="python")
        st.caption("这就是神经网络眼中的“用户”。")

# ==========================================
# 模块 4: 面试模拟 (游戏化升级)
# ==========================================
elif module == "4. 面试模拟 (实战)":
    st.title("⚔️ 模块 4: 推荐系统面试模拟")
    st.markdown("不要死记硬背。点击问题，先思考，再查看**大厂标准答案**。")

    # 封装一个显示问题的函数
    def show_qna(question, answer, key):
        st.markdown(f"#### ❓ {question}")
        # 使用 checkbox 模拟“点击查看答案”
        if st.checkbox("查看解析", key=key):
            st.success("✅ **面试官期望的回答点：**")
            st.markdown(answer)
        st.markdown("---")

    show_qna(
        "如何处理冷启动 (Cold Start) 问题？",
        """
        1. **利用热门榜单**：新用户进来先推热门 (Top-K)。
        2. **利用内容特征 (Content-based)**：如果有用户的注册信息（年龄、性别）或物品的标签，直接做相似度匹配。
        3. **利用探索与利用 (Exploit & Explore)**：使用 **MAB (多臂老虎机)** 算法，给新物品少量的流量进行测试。
        """,
        "q1"
    )
    
    show_qna(
        "协同过滤 (CF) 和 矩阵分解 (MF) 有什么本质区别？",
        """
        * **CF (记忆)**：像查字典。直接找历史行为相似的人。*缺点：存不下大矩阵，稀疏时效果差。*
        * **MF (泛化)**：像做阅读理解。把人及物映射到隐向量空间，通过向量内积计算分数。*优点：能预测未见过的交互，泛化能力强。*
        """,
        "q2"
    )
    
    show_qna(
        "为什么 DeepFM 比 LR (逻辑回归) 效果好？",
        """
        * **LR** 只能学到一阶特征（线性的），必须人工做大量的特征工程（比如手动组合“啤酒+尿布”）。
        * **DeepFM** 结合了 FM 和 DNN：
            1.  **FM部分**：自动学习二阶特征交叉。
            2.  **DNN部分**：学习高阶、非线性的特征组合。
        """,
        "q3"
    )

    st.info("💡 提示：面试中如果能画出模块 3 中的架构图，通常会加分！")