import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import deque
import math
import csv 
import altair as alt 
import matplotlib.font_manager as fm

font_path = r"instructional simulation/CN_FONT.ttf"   # 字体路径（相对路径）
fm.fontManager.addfont(font_path)            # 注册字体
plt.rcParams['font.family'] = 'Noto Sans SC' # 设置为思源黑体
plt.rcParams['axes.unicode_minus'] = False   # 避免负号乱码

# 设置页面编码
st.set_page_config(layout="wide", page_title="Athlete Physiology Simulator")

# 添加HTML编码声明
st.markdown("""
    <meta charset='utf-8'/>
    <style>
        .main .block-container {
            font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# ① 核心生理学模型（保持不变）
# --------------------------------------------------------------------------

def get_target_power(t, chosen_mode, athlete_dict, interval_state, dt=0.05):
    """获取当前时间的目标功率"""
    
    modes = {
        "恢复跑": {"intensity": 0.5}, "轻松跑": {"intensity": 0.65},
        "节奏跑": {"intensity": 0.8}, "阈值跑": {"intensity": 0.88},
        "间歇跑": {"intensity": 0.95}, "冲刺": {"intensity": 1.0},
    }
    
    if chosen_mode == "间歇跑":
        work_inten = athlete_dict["间歇-工作强度"]
        work_dur = athlete_dict["间歇-工作时间(s)"]
        rest_inten = athlete_dict["间歇-休息强度"]
        rest_dur = athlete_dict["间歇-休息时间(s)"]
        
        interval_pattern = [(work_dur, work_inten), (rest_dur, rest_inten)]
        
        if interval_state["index"] >= len(interval_pattern):
            interval_state["index"] = 0
            
        dur, inten = interval_pattern[interval_state["index"]]
        
        if interval_state["elapsed"] >= dur:
            interval_state["index"] = (interval_state["index"] + 1) % len(interval_pattern)
            interval_state["elapsed"] = 0.0
            dur, inten = interval_pattern[interval_state["index"]]
            
        interval_state["elapsed"] += dt 
        return inten
    
    return modes.get(chosen_mode, {"intensity": 0.6})["intensity"]

def vo2_ss_for_power(P, athlete_dict): 
    CP = athlete_dict["临界功率(CP)"]
    VO2_fast_ss = min(1.0, P)
    VO2_slow_ss = 0.0
    if P > CP:
        VO2_slow_ss = min(0.2, (P - CP) * 0.5)
    return VO2_fast_ss, VO2_slow_ss

def sim_step(state, athlete_dict, interval_state, dt=0.05):
    """核心模拟步骤，更新 state 字典"""
    
    # 1. 获取动态参数
    CP = athlete_dict["临界功率(CP)"]
    W_prime = athlete_dict["无氧储备(W')"]
    tau_fast = 20.0
    tau_slow = 300.0
    tau_w_rec = 300.0
    
    # 2. 目标功率
    P_target = get_target_power(state["t"], athlete_dict["运动类型"], athlete_dict, interval_state, dt)
    state["P_target"] = P_target
    
    # 3. 实际功率 (带疲劳)
    P_current = state["power"]
    W_rem_norm = state["Wrem"] / W_prime if W_prime > 0 else 0
    fatigue_factor = 0.2 + 0.8 * (W_rem_norm / 0.2) if W_rem_norm < 0.2 else 1.0
    P_effective_target = P_target * fatigue_factor
    dP = (P_effective_target - P_current) / 3.0
    P = P_current + dP * dt
    state["power"] = max(0.0, min(1.0, P))
    
    # 4. VO₂ 动态
    VO2_fast_ss, VO2_slow_ss = vo2_ss_for_power(P, athlete_dict)
    state["VO2_fast"] += (VO2_fast_ss - state["VO2_fast"]) / tau_fast * dt
    state["VO2_slow"] += (VO2_slow_ss - state["VO2_slow"]) / tau_slow * dt
    state["VO2_total"] = state["VO2_fast"] + state["VO2_slow"]
    
    # 5. PCr
    p_thresh = 0.6 
    k_depl = 0.02 + 0.5 * max(0.0, P - p_thresh)
    pcr_recovery = (1.0 - state["PCr"]) * state["VO2_total"] / 45.0
    state["PCr"] += (-k_depl * P * state["PCr"] + pcr_recovery) * dt
    
    # 6. 乳酸
    prod = 1.5 * max(0.0, P - CP) * (1.0 - state["PCr"])
    k_clear = 0.0017
    state["Lac"] += (prod - k_clear * state["Lac"]) * dt
    
    # 7. ATP
    k_resyn = 0.12 * (state["PCr"] + state["VO2_total"])
    k_use = 0.06 * state["power"]
    state["ATP"] += (k_resyn * (1 - state["ATP"]) - k_use * state["ATP"]) * dt
    
    # 8. W'
    if P > CP:
        state["Wrem"] -= (P - CP) * 0.01 * dt
    else:
        VO2_recovery_factor = max(0.01, (1.0 - state["VO2_total"]) / (1.0 - CP))
        VO2_recovery_factor = min(1.0, VO2_recovery_factor)
        state["Wrem"] += (W_prime - state["Wrem"]) / tau_w_rec * VO2_recovery_factor * 2.0 * dt
        
    state["t"] += dt
    
    # 9. 限制
    for k in ["ATP", "PCr", "VO2_total", "VO2_fast", "VO2_slow"]:
        state[k] = max(0.0, min(1.0, state[k]))
    state["Lac"] = max(0.0, state["Lac"])
    state["Wrem"] = max(0.0, min(W_prime, state["Wrem"]))

    return state


# --------------------------------------------------------------------------
# ② Streamlit 绘图函数 (已优化)
# --------------------------------------------------------------------------

try:
    plt.rcParams['font.sans-serif'] = ['HeiTi', 'Heiti TC', 'PingFang SC', 'STHeiti', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def create_plot_fig(buffers, athlete_dict):
    """固定 Matplotlib 尺寸，消除布局抖动"""
    CP_dynamic = athlete_dict["临界功率(CP)"]
    
    fig, ax = plt.subplots(figsize=(8, 6)) 
    fig.patch.set_facecolor('#FFFFFF') 
    ax.set_facecolor('#FFFFFF')

    if not buffers["time"]:
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1.1)
        ax.set_title("等待开始...")
        return fig
    
    t = list(buffers["time"])
    tmin, tmax = t[0], t[-1]
    
    ax.plot(buffers["time"], buffers["power"], label="Power(强度)", color="#C832C8", zorder=5)
    ax.plot(buffers["time"], buffers["vo2"], label="VO2(总有氧)", color="#3232C8", zorder=4)
    ax.plot(buffers["time"], buffers["pcr"], label="PCr(ATP-CP)", color="#E67814", zorder=3)
    ax.plot(buffers["time"], buffers["lac"], label="Lactate(无氧糖解)", color="#14A03C", zorder=2)
    ax.plot(buffers["time"], buffers["atp"], label="ATP(剩余量)", color="#FF0000", zorder=6)
    
    ax.axhline(y=CP_dynamic, color='red', linestyle='--', linewidth=1, zorder=1)
    ax.fill_between(t, CP_dynamic, 1.1, color='red', alpha=0.1, zorder=0)
    ax.text(tmin, CP_dynamic + 0.02, f"CP = {CP_dynamic:.2f}", color='red')

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("相对值")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(tmin, max(tmax, tmin + 10))
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    
    return fig

def create_bar_chart(state, athlete_dict):
    """固定 Altair 尺寸，无频闪"""
    W_prime = athlete_dict["无氧储备(W')"]
    
    # 计算贡献值
    c_atp = state["PCr"] * 2.0
    W_used = W_prime - state["Wrem"]
    c_anaer_glyco = max(0.0, W_used * 10)
    c_aero_glyco = max(0.0, state["power"] * state["VO2_total"] - state["VO2_total"] * 0.3)
    c_aero = state["VO2_total"] * 1.5
    c_atp = max(0.1, c_atp)
    
    sumc = c_atp + c_anaer_glyco + c_aero_glyco + c_aero + 1e-6
    
    vals = [c_atp/sumc, c_anaer_glyco/sumc, c_aero_glyco/sumc, c_aero/sumc]
    labels = ["ATP-PCr", "无氧糖酵解(W')", "有氧糖酵解", "总有氧(脂肪+)"]
    colors = ['#E67814', '#C832C8', '#14A03C', '#3232C8']
    
    # 1. 创建 DataFrame
    df = pd.DataFrame({
        'labels': labels,
        'values': vals,
        'colors': colors,
        'text_labels': [f"{v*100:.0f}%" for v in vals]
    })
    
    # 2. 创建图表
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('labels', sort=None, title=None),
        x=alt.X('values', title='能量贡献占比', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('colors', scale=None) 
    ).properties(
        title='实时能量系统贡献',
        width=300, 
        height=300
    )
    
    text = chart.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text='text_labels',
        color=alt.value('black')
    )

    return (chart + text).interactive()

def create_csv_content(buffers):
    if not buffers["time"]:
        return "没有可导出的数据"
    df = pd.DataFrame({
        "Time(s)": buffers["time"], "Power": buffers["power"],
        "VO2_Total": buffers["vo2"], "PCr": buffers["pcr"],
        "Lactate": buffers["lac"], "ATP": buffers["atp"],
    })
    return df.to_csv(index=False).encode('utf-8-sig')


# --------------------------------------------------------------------------
# ③ Streamlit 应用主逻辑 (已修改)
# --------------------------------------------------------------------------
def initialize_state():
    """初始化所有 session_state 变量"""
    if 'initialized' not in st.session_state:
        st.session_state.athlete_dict = {
            "体重(kg)": 70.0, "肌肉比例": 0.40, "快肌比例": 0.45,
            "最大摄氧量(VO2max)": 1.0, "最大功率(Pmax)": 1.0,
            "临界功率(CP)": 0.6, "无氧储备(W')": 0.2,
            "运动时间(min)": 10.0, "运动类型": "节奏跑",
            "间歇-工作强度": 0.95, "间歇-工作时间(s)": 30.0,
            "间歇-休息强度": 0.55, "间歇-休息时间(s)": 60.0,
        }
        
        st.session_state.state = {
            "t": 0.0, "ATP": 1.0, "PCr": 1.0, "VO2_fast": 0.0, "VO2_slow": 0.0,
            "VO2_total": 0.0, "Lac": 0.0, "Wrem": 0.2, "power": 0.0, "P_target": 0.0,
        }
        
        st.session_state.interval_state = {"index": 0, "elapsed": 0.0}
        
        max_points = int(600 / 0.05)
        st.session_state.buffers = {
            "time": deque(maxlen=max_points), "power": deque(maxlen=max_points),
            "vo2": deque(maxlen=max_points), "pcr": deque(maxlen=max_points),
            "lac": deque(maxlen=max_points), "atp": deque(maxlen=max_points),
        }
        
        st.session_state.running = False
        st.session_state.initialized = True
        st.session_state.state["Wrem"] = st.session_state.athlete_dict["无氧储备(W')"]
        
        st.session_state.speed_multiplier = 1.0
        st.session_state.loop_control = {
            "real_time_start": 0.0,
            "sim_time_start": 0.0
        }


# --- 主应用运行 ---
st.set_page_config(layout="wide", page_title="虚拟运动员模拟器")
initialize_state()

# --- 侧边栏 UI (左侧面板) ---
with st.sidebar:
    st.title("虚拟运动员属性")
    
    st.session_state.speed_multiplier = st.select_slider(
        "模拟倍速 (1x = 实时)",
        options=[1.0, 2.0, 5.0, 10.0, 20.0],
        value=st.session_state.speed_multiplier
    )
    
    athlete_keys = list(st.session_state.athlete_dict.keys())
    
    # 🌟 移除：不再需要非生理学参数，只保留需要的
    filtered_keys = [
        "体重(kg)", "肌肉比例", "快肌比例", "最大摄氧量(VO2max)", 
        "最大功率(Pmax)", "临界功率(CP)", "无氧储备(W')", 
        "运动时间(min)", "运动类型", 
        "间歇-工作强度", "间歇-工作时间(s)", 
        "间歇-休息强度", "间歇-休息时间(s)"
    ]
    
    for key in filtered_keys:
        value = st.session_state.athlete_dict[key]
        
        if key == "运动类型":
            modes_list = ["恢复跑", "轻松跑", "节奏跑", "阈值跑", "间歇跑", "冲刺"]
            idx = modes_list.index(value) if value in modes_list else 0
            st.session_state.athlete_dict[key] = st.selectbox(key, modes_list, index=idx)
        
        elif "间歇-" in key:
            is_disabled = st.session_state.athlete_dict["运动类型"] != "间歇跑"
            st.session_state.athlete_dict[key] = st.number_input(key, value=value, format="%.2f", disabled=is_disabled)
        
        elif isinstance(value, float):
            st.session_state.athlete_dict[key] = st.number_input(key, value=value, format="%.2f")

    st.subheader("--- 实时状态 ---")
    st.text(f"t = {st.session_state.state['t']:.1f}s")
    st.text(f"Target P = {st.session_state.state['P_target']:.2f}")
    st.text(f"Actual P = {st.session_state.state['power']:.2f}")
    st.text(f"W' 剩余: {st.session_state.state['Wrem']:.3f}")

    st.subheader("--- 控制台 ---")
    col1, col2 = st.columns(2)
    
    # 🌟 修改：Play -> 运行
    if col1.button("运行", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.session_state.loop_control['real_time_start'] = time.time()
        st.session_state.loop_control['sim_time_start'] = st.session_state.state['t']
        st.rerun()

    if col2.button("Pause", use_container_width=True):
        st.session_state.running = False
        st.rerun()

    if st.button("Reset", use_container_width=True):
        st.session_state.running = False
        st.session_state.pop('initialized') 
        initialize_state()
        st.rerun()

    st.download_button(
        label="保存数据 (CSV)",
        data=create_csv_content(st.session_state.buffers),
        file_name=f"athlete_sim_data_{int(time.time())}.csv",
        mime='text/csv',
        use_container_width=True
    )

# --- 主面板 (右侧) ---
st.title("动态生理变化")

col1, col2 = st.columns([3, 1]) 

plot_fig = create_plot_fig(st.session_state.buffers, st.session_state.athlete_dict)
bar_chart = create_bar_chart(st.session_state.state, st.session_state.athlete_dict)

with col1:
    st.pyplot(plot_fig)

with col2:
    st.altair_chart(bar_chart)


# --------------------------------------------------------------------------
# ④ 模拟 "Game Loop" (保持不变)
# --------------------------------------------------------------------------
if st.session_state.running:
    
    speed = st.session_state.speed_multiplier
    dt = 0.05
    loop_control = st.session_state.loop_control
    
    real_time_elapsed = time.time() - loop_control['real_time_start']
    target_sim_time = loop_control['sim_time_start'] + (real_time_elapsed * speed)
    current_sim_time = st.session_state.state['t']
    
    steps_to_run = int((target_sim_time - current_sim_time) / dt)
    
    if steps_to_run > 0:
        buffers = st.session_state.buffers
        for _ in range(steps_to_run):
            if st.session_state.state['t'] > st.session_state.athlete_dict["运动时间(min)"] * 60:
                st.session_state.running = False
                break
            
            st.session_state.state = sim_step(
                st.session_state.state, 
                st.session_state.athlete_dict, 
                st.session_state.interval_state,
                dt=dt
            )
            
            state = st.session_state.state
            buffers["time"].append(state["t"])
            buffers["power"].append(state["power"])
            buffers["vo2"].append(state["VO2_total"])
            buffers["pcr"].append(state["PCr"])
            buffers["lac"].append(state["Lac"])
            buffers["atp"].append(state["ATP"])

    if st.session_state.running:
        time.sleep(0.1) 
        st.rerun()
    else:
        st.rerun()