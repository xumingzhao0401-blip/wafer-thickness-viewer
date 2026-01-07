# -*- coding: utf-8 -*-
"""
Wafer Thickness Viewer (Streamlit) - Dataset Manager Version

改进点（解决“CSV 与生成器数据冲突”）：
1) 引入「数据集管理器」：每片晶圆数据都作为一个 dataset（有名字）存入 st.session_state.datasets
2) 普通模式通过下拉框选择当前显示的数据集；CSV 导入 / 生成器保存都只会“新增数据集”，不会覆盖彼此
3) 保留你原有逻辑：缓存、统计、3D 视角控制、标签开关、生成器预览图、下载模板等
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="晶圆厚度可视化",
    page_icon="🟢",
    layout="wide",
)

# ============================================================
# Dataset Manager (NEW)
# ============================================================

def _init_dataset_store():
    if "datasets" not in st.session_state:
        # name -> dict(df, wafer_inch, cmap, spec_upper, spec_lower, created_at)
        st.session_state.datasets = {}
    if "active_ds" not in st.session_state:
        st.session_state.active_ds = None


def _unique_name(base: str) -> str:
    base = (base or "Wafer").strip()
    name = base
    i = 2
    while name in st.session_state.datasets:
        name = f"{base} ({i})"
        i += 1
    return name


def register_dataset(
    name: str,
    df: pd.DataFrame,
    wafer_inch: float,
    cmap: str = "viridis",
    spec_upper: Optional[float] = None,
    spec_lower: Optional[float] = None,
):
    _init_dataset_store()
    name = _unique_name(name)
    st.session_state.datasets[name] = {
        "df": df.copy(),
        "wafer_inch": float(wafer_inch),
        "cmap": cmap,
        "spec_upper": spec_upper,
        "spec_lower": spec_lower,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.active_ds = name


def get_active_dataset() -> Tuple[Optional[str], Optional[dict]]:
    _init_dataset_store()
    name = st.session_state.active_ds
    if not name:
        return None, None
    return name, st.session_state.datasets.get(name)


def rename_dataset(old_name: str, new_name: str):
    _init_dataset_store()
    if old_name not in st.session_state.datasets:
        return
    new_name = (new_name or "").strip()
    if not new_name:
        return
    if new_name == old_name:
        return
    new_name = _unique_name(new_name)
    st.session_state.datasets[new_name] = st.session_state.datasets.pop(old_name)
    st.session_state.active_ds = new_name


def delete_dataset(name: str):
    _init_dataset_store()
    st.session_state.datasets.pop(name, None)
    st.session_state.active_ds = next(iter(st.session_state.datasets), None)


# ============================================================
# Utils / Core math (Cached)
# ============================================================

def build_grids(radius_mm: float, grid_res: int) -> Tuple[np.ndarray, np.ndarray]:
    x_lin = np.linspace(-radius_mm, radius_mm, grid_res)
    y_lin = np.linspace(-radius_mm, radius_mm, grid_res)
    return np.meshgrid(x_lin, y_lin)


@st.cache_data
def cached_idw_interpolation(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    radius_mm: float,
    grid_res: int,
    power: float = 2.0,
    eps: float = 1e-6,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if len(xs) < 3:
        return None, None, None

    grid_x, grid_y = build_grids(radius_mm, grid_res)

    gx = grid_x[..., np.newaxis]
    gy = grid_y[..., np.newaxis]
    px = xs[np.newaxis, np.newaxis, :]
    py = ys[np.newaxis, np.newaxis, :]

    dx = gx - px
    dy = gy - py
    dist = np.sqrt(dx**2 + dy**2) + eps

    weights = 1.0 / (dist**power)
    w_sum = np.sum(weights, axis=2)
    t_weighted_sum = np.sum(weights * ts[np.newaxis, np.newaxis, :], axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        grid_t = t_weighted_sum / w_sum

    mask = (grid_x**2 + grid_y**2) <= (radius_mm**2)
    grid_t = np.where(mask, grid_t, np.nan)

    return grid_x, grid_y, grid_t


@st.cache_data
def compute_statistics(ts: np.ndarray) -> pd.DataFrame:
    ts = np.asarray(ts, dtype=float)
    ts = ts[~np.isnan(ts)]

    n = int(ts.size)
    if n == 0:
        return pd.DataFrame(columns=["特征名", "公式", "计算结果", "单位"])

    mean = float(np.mean(ts))
    std = float(np.std(ts, ddof=1)) if n > 1 else 0.0

    min_v = float(np.min(ts))
    max_v = float(np.max(ts))
    p2p = max_v - min_v

    safe_mean = mean if mean != 0 else 1e-9
    safe_sum = (max_v + min_v) if (max_v + min_v) != 0 else 1e-9

    cov = (std / safe_mean) * 100.0
    wiw_nu = (p2p / safe_sum) * 100.0
    range_mean = (p2p / safe_mean) * 100.0
    max_dev = (float(np.max(np.abs(ts - mean))) / safe_mean) * 100.0
    u3 = (3.0 * std / safe_mean) * 100.0
    u6 = (6.0 * std / safe_mean) * 100.0

    rows = [
        ("点数 N", "N", n, ""),
        ("平均值 (Mean)", "μ = (1/N)·Σxi", mean, ""),
        ("标准差 (Std, 1σ)", "σ = sqrt( Σ(xi-μ)^2 / (N-1) )", std, ""),
        ("变异系数 (CoV)", "CoV = σ/μ × 100%", cov, "%"),
        ("最小值 (Min)", "min(xi)", min_v, ""),
        ("最大值 (Max)", "max(xi)", max_v, ""),
        ("峰-峰值 (Peak-to-Peak)", "P-P = max - min", p2p, ""),
        ("WIWNU（常用均匀度）", "(max-min)/(max+min) × 100%", wiw_nu, "%"),
        ("Range/Mean", "(max-min)/μ × 100%", range_mean, "%"),
        ("最大偏差 (Max Dev.)", "max(|xi-μ|)/μ × 100%", max_dev, "%"),
        ("3σ 均匀度", "3σ/μ × 100%", u3, "%"),
        ("6σ 均匀度", "6σ/μ × 100%", u6, "%"),
    ]

    df = pd.DataFrame(rows, columns=["特征名", "公式", "计算结果", "单位"])
    df["计算结果"] = df["计算结果"].apply(
        lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    )
    return df


@st.cache_data
def load_csv(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"读取 CSV 失败：{e}")
        return None

    lower_map = {c.lower().strip(): c for c in df.columns}
    required = ["x", "y", "thickness"]
    missing = [k for k in required if k not in lower_map]
    if missing:
        st.error(f"CSV 缺少列：{missing}")
        return None

    df2 = df[[lower_map["x"], lower_map["y"], lower_map["thickness"]]].copy()
    df2.columns = ["x", "y", "thickness"]
    for c in required:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2 = df2.dropna(subset=["x", "y", "thickness"]).reset_index(drop=True)
    if df2.empty:
        st.error("CSV 解析后没有有效数据。")
        return None
    return df2


def wafer_radius_mm(wafer_inch: float) -> float:
    return float(wafer_inch) * 25.4 / 2.0


def circle_boundary_trace(radius_mm: float, n: int = 361) -> go.Scatter:
    theta = np.linspace(0, 2 * np.pi, n)
    x = radius_mm * np.cos(theta)
    y = radius_mm * np.sin(theta)
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Wafer Boundary",
        showlegend=False,
        line=dict(width=3, color="black"),
    )


# ============================================================
# Drawing Functions
# ============================================================

def make_top_view_heatmap(
    df: pd.DataFrame,
    radius_mm: float,
    cmap: str,
    grid_res: int,
    show_labels: bool = True,
) -> go.Figure:
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    ts = df["thickness"].to_numpy(dtype=float)

    grid_x, grid_y, grid_t = cached_idw_interpolation(xs, ys, ts, radius_mm, grid_res)

    fig = go.Figure()
    if grid_t is not None:
        fig.add_trace(
            go.Heatmap(
                z=grid_t,
                x=grid_x[0, :],
                y=grid_y[:, 0],
                colorscale=cmap,
                colorbar=dict(title="Thickness", thickness=18, len=0.85),
                hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>T=%{z:.4f}<extra></extra>",
            )
        )

    fig.add_trace(circle_boundary_trace(radius_mm))

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=8, color="black", line=dict(width=1, color="white")),
            showlegend=False,
            hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>T=%{text}<extra></extra>",
            text=[f"{v:.4f}" for v in ts],
        )
    )

    if show_labels:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="text",
                text=[f"{v:.4f}" for v in ts],
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
                textfont=dict(color="black", size=10, family="Arial"),
            )
        )

    fig.update_layout(
        title="Top View Heatmap with Data Labels",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            title="X (mm)",
            scaleanchor="y",
            scaleratio=1,
            range=[-radius_mm * 1.05, radius_mm * 1.05],
        ),
        yaxis=dict(title="Y (mm)", range=[-radius_mm * 1.05, radius_mm * 1.05]),
        height=800,
    )
    return fig


def make_3d_surface(
    df: pd.DataFrame,
    radius_mm: float,
    cmap: str,
    grid_res: int,
    spec_upper: Optional[float],
    spec_lower: Optional[float],
    camera_eye: Optional[Tuple[float, float, float]] = None,
) -> go.Figure:
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    ts = df["thickness"].to_numpy(dtype=float)

    grid_x, grid_y, grid_t = cached_idw_interpolation(xs, ys, ts, radius_mm, grid_res)

    fig = go.Figure()

    if grid_t is not None:
        fig.add_trace(
            go.Surface(
                x=grid_x,
                y=grid_y,
                z=grid_t,
                colorscale=cmap,
                colorbar=dict(title="Thickness", thickness=18, len=0.85),
                hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>T=%{z:.4f}<extra></extra>",
            )
        )
        mask = ~np.isnan(grid_t)

        if spec_upper is not None and np.isfinite(spec_upper):
            z_up = np.where(mask, spec_upper, np.nan)
            fig.add_trace(
                go.Surface(
                    x=grid_x,
                    y=grid_y,
                    z=z_up,
                    opacity=0.25,
                    showscale=False,
                    name="USL",
                    colorscale=[[0, "red"], [1, "red"]],
                )
            )
        if spec_lower is not None and np.isfinite(spec_lower):
            z_lo = np.where(mask, spec_lower, np.nan)
            fig.add_trace(
                go.Surface(
                    x=grid_x,
                    y=grid_y,
                    z=z_lo,
                    opacity=0.25,
                    showscale=False,
                    name="LSL",
                    colorscale=[[0, "red"], [1, "red"]],
                )
            )

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=ts,
            mode="markers",
            marker=dict(size=4, color="black"),
            name="量测点",
        )
    )

    scene = dict(
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Thickness",
        aspectmode="data",
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
    )
    if camera_eye is not None:
        scene["camera"] = dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))

    fig.update_layout(
        title="Wafer Thickness 3D Distribution",
        scene=scene,
        margin=dict(l=0, r=0, t=50, b=0),
        height=900,
    )
    return fig


# ============================================================
# New Function: Pattern Preview Plot
# ============================================================

def plot_pattern_preview(df: pd.DataFrame, radius_mm: float) -> go.Figure:
    """盲点模式预览：晶圆轮廓 + 点位 + 序号"""
    fig = go.Figure()
    fig.add_trace(circle_boundary_trace(radius_mm))
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers+text",
            marker=dict(size=12, color="red"),
            text=[str(i) for i in df.index],
            textposition="top center",
            textfont=dict(size=14, color="red", family="Arial Black"),
            name="Points",
        )
    )

    fig.update_layout(
        title="点位分布预览 (Preview)",
        width=800,
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            range=[-radius_mm * 1.1, radius_mm * 1.1],
            scaleanchor="y",
            scaleratio=1,
            zeroline=True,
            showgrid=True,
        ),
        yaxis=dict(
            range=[-radius_mm * 1.1, radius_mm * 1.1],
            zeroline=True,
            showgrid=True,
        ),
        showlegend=False,
        hovermode="closest",
    )
    return fig


# ============================================================
# Logic: Generator
# ============================================================

def generate_pattern_coords(pattern_type, radius_mm, edge_exclude_mm, **kwargs) -> List[Tuple[float, float]]:
    effective_r = max(0, radius_mm - edge_exclude_mm)
    points = []

    if pattern_type == "十字交叉 (Cross)":
        n_per_arm = kwargs.get("points_per_arm", 3)
        points.append((0.0, 0.0))
        if effective_r > 0:
            radii = np.linspace(0, effective_r, n_per_arm + 1)[1:]
            for r in radii:
                points.extend([(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)])

    elif pattern_type == "同心圆 (Concentric)":
        n_rings = kwargs.get("n_rings", 3)
        pts_per_ring = kwargs.get("pts_per_ring", 8)
        points.append((0.0, 0.0))
        if effective_r > 0:
            radii = np.linspace(0, effective_r, n_rings + 1)[1:]
            for r in radii:
                angles = np.linspace(0, 2 * np.pi, pts_per_ring, endpoint=False)
                for ang in angles:
                    points.append((r * np.cos(ang), r * np.sin(ang)))

    elif pattern_type == "均匀网格 (Grid)":
        step = kwargs.get("grid_step", 30.0)
        xs = np.arange(0, effective_r + 0.1, step)
        xs = np.concatenate((-xs[:0:-1], xs))
        ys = xs.copy()
        for x in xs:
            for y in ys:
                if (x**2 + y**2) <= effective_r**2:
                    points.append((x, y))

    return points


def blind_mode_ui():
    _init_dataset_store()

    st.markdown("### 🛠️ 坐标生成器 (Generator Mode)")
    st.caption("无需手动点击。选择测量图案，自动生成标准坐标。**没有数据的点请直接留空，保存时会自动忽略。**")

    col_ctrl, col_data = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.subheader("1. 晶圆设置")
        wafer_inch = st.selectbox("尺寸 (Inch)", [4, 6, 8, 12], index=2, key="gen_inch")
        radius_mm = wafer_radius_mm(wafer_inch)
        ee_mm = st.number_input("Edge Exclusion (mm)", value=3.0, min_value=0.0, key="gen_ee")

        st.subheader("2. 图案选择")
        pat_type = st.radio("生成方式", ["十字交叉 (Cross)", "同心圆 (Concentric)", "均匀网格 (Grid)"], key="gen_pat")

        gen_params = {}
        if pat_type == "十字交叉 (Cross)":
            gen_params["points_per_arm"] = st.slider("每臂点数", 1, 15, 3, key="gen_arm")
        elif pat_type == "同心圆 (Concentric)":
            gen_params["n_rings"] = st.slider("圈数", 1, 10, 3, key="gen_rings")
            gen_params["pts_per_ring"] = st.slider("每圈点数", 4, 32, 8, step=4, key="gen_ppr")
        elif pat_type == "均匀网格 (Grid)":
            gen_params["grid_step"] = st.number_input("网格间距 (mm)", value=30.0, min_value=5.0, key="gen_step")

        st.subheader("3. 数据集命名")
        default_name = f"GEN {pat_type.split(' ')[0]} {datetime.now().strftime('%H%M%S')}"
        st.text_input("保存为数据集名称", value=default_name, key="gen_ds_name")

        if st.button("🔄 生成坐标表", type="primary", use_container_width=True, key="gen_build"):
            pts = generate_pattern_coords(pat_type, radius_mm, ee_mm, **gen_params)
            df = pd.DataFrame(pts, columns=["x", "y"])
            df["thickness"] = np.nan
            st.session_state.gen_df = df
            st.rerun()

    with col_data:
        st.subheader("4. 数据录入")

        if "gen_df" not in st.session_state:
            st.info("👈 请先配置参数并点击生成")
            return

        df_curr = st.session_state.gen_df

        with st.expander("👁️ 查看点位分布示意图 (Preview)", expanded=True):
            preview_fig = plot_pattern_preview(df_curr, radius_mm)
            st.plotly_chart(preview_fig, use_container_width=True)

        st.markdown(f"**点数：{len(df_curr)}**。请填入厚度（无数据点留空）：")

        edited_df = st.data_editor(
            df_curr,
            column_config={
                "x": st.column_config.NumberColumn(format="%.2f", disabled=True),
                "y": st.column_config.NumberColumn(format="%.2f", disabled=True),
                "thickness": st.column_config.NumberColumn("Thickness", format="%.4f", required=False),
            },
            use_container_width=True,
            height=300,
            key="data_editor_gen",
        )

        c_act = st.columns([1, 2])

        with c_act[0]:
            if st.button("✅ 保存为数据集并分析", type="primary", use_container_width=True, key="gen_save"):
                final_df = edited_df.dropna(subset=["thickness"])
                if final_df.empty:
                    st.error("所有点厚度均为空！")
                else:
                    register_dataset(
                        name=st.session_state.get("gen_ds_name", "Generator"),
                        df=final_df,
                        wafer_inch=wafer_inch,
                        cmap="viridis",
                        spec_upper=None,
                        spec_lower=None,
                    )
                    st.success(
                        f"已保存为数据集：{st.session_state.active_ds}（有效点 {len(final_df)}）！请到「普通模式」选择查看。"
                    )
                    st.rerun()

        with c_act[1]:
            csv = edited_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 下载 CSV 模板", csv, "template.csv", "text/csv", key="gen_dl")


# ============================================================
# Normal Mode UI (Dataset-driven)
# ============================================================

def normal_mode_ui():
    _init_dataset_store()

    st.markdown("### 普通模式")
    st.info("坐标系说明：CSV 中 (x,y) 以晶圆中心为 (0,0)，四象限坐标；单位 mm。", icon="ℹ️")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("#### 数据集管理")

        names = list(st.session_state.datasets.keys())
        if names:
            default_idx = names.index(st.session_state.active_ds) if st.session_state.active_ds in names else 0
            chosen = st.selectbox("选择要显示的数据集", names, index=default_idx, key="ds_select")
            st.session_state.active_ds = chosen
            ds_name, ds = get_active_dataset()
        else:
            ds_name, ds = None, None
            st.info("当前还没有数据集：你可以导入 CSV 或在生成器中保存为数据集。", icon="🧊")

        if ds_name and ds:
            c1, c2 = st.columns([1, 1])
            with c1:
                new_name = st.text_input("重命名当前数据集", value=ds_name, key="ds_rename_txt")
                if st.button("✏️ 应用重命名", use_container_width=True, key="ds_rename_btn"):
                    rename_dataset(ds_name, new_name)
                    st.rerun()
            with c2:
                st.caption(f"创建时间：{ds.get('created_at','-')}")
                if st.button("🗑️ 删除当前数据集", use_container_width=True, key="ds_del_btn"):
                    delete_dataset(ds_name)
                    st.rerun()

        st.markdown("---")
        st.markdown("#### 导入新的 CSV 作为数据集")
        uploaded = st.file_uploader("选择 CSV 数据文件", type=["csv"], key="normal_uploader")

        import_inch = st.selectbox("导入时晶圆尺寸（英寸）", [4, 6, 8, 12], index=2, key="import_wafer")
        default_import_name = ""
        if uploaded is not None:
            default_import_name = f"{uploaded.name} {datetime.now().strftime('%H%M%S')}"
        import_name = st.text_input("新数据集名称", value=default_import_name, key="import_name")

        if st.button(
            "➕ 导入为新数据集",
            type="primary",
            use_container_width=True,
            disabled=(uploaded is None),
            key="import_btn",
        ):
            df_new = load_csv(uploaded)
            if df_new is not None:
                register_dataset(
                    name=import_name or (uploaded.name if uploaded else "CSV"),
                    df=df_new,
                    wafer_inch=import_inch,
                    cmap="viridis",
                    spec_upper=None,
                    spec_lower=None,
                )
                st.success(f"已导入数据集：{st.session_state.active_ds}")
                st.rerun()

        st.markdown("---")
        st.markdown("#### 绘图参数（作用于当前选中数据集）")

        if ds is None:
            st.info("请先选择/创建一个数据集。")
            return

        # defaults from dataset meta
        wafer_inch_default = ds.get("wafer_inch", 8)
        cmap_default = ds.get("cmap", "viridis")
        spec_up_default = "" if ds.get("spec_upper") is None else str(ds.get("spec_upper"))
        spec_lo_default = "" if ds.get("spec_lower") is None else str(ds.get("spec_lower"))

        wafer_inch = st.selectbox(
            "晶圆尺寸（英寸）",
            [4, 6, 8, 12],
            index=[4, 6, 8, 12].index(int(wafer_inch_default)),
            key="normal_wafer",
        )
        radius_mm = wafer_radius_mm(wafer_inch)

        cmaps = ["viridis", "plasma", "inferno", "magma", "coolwarm", "cividis"]
        cmap = st.selectbox(
            "色图 (colormap)",
            cmaps,
            index=cmaps.index(cmap_default) if cmap_default in cmaps else 0,
            key="normal_cmap",
        )

        spec_cols = st.columns(2)
        with spec_cols[0]:
            spec_upper_txt = st.text_input("上限 SPEC（可选）", value=spec_up_default, key="normal_spec_up_txt")
        with spec_cols[1]:
            spec_lower_txt = st.text_input("下限 SPEC（可选）", value=spec_lo_default, key="normal_spec_lo_txt")

        def _parse(s):
            try:
                return float(s) if s.strip() else None
            except Exception:
                return None

        spec_upper = _parse(spec_upper_txt)
        spec_lower = _parse(spec_lower_txt)

        # write back to dataset meta (each dataset remembers its own)
        ds["wafer_inch"] = wafer_inch
        ds["cmap"] = cmap
        ds["spec_upper"] = spec_upper
        ds["spec_lower"] = spec_lower

        grid_res = st.slider("插值网格分辨率（越高越细，越慢）", 120, 360, 220, 10, key="normal_grid")
        show_labels = st.checkbox("顶视图显示厚度数值标签", value=True, key="normal_labels")

        st.markdown("#### 统计结果")
        stats_df = compute_statistics(ds["df"]["thickness"].to_numpy(dtype=float))
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### 点表与顶视图")

        df = ds["df"]
        st.dataframe(
            df.rename(columns={"x": "X (mm)", "y": "Y (mm)", "thickness": "Thickness"}),
            use_container_width=True,
            hide_index=True,
        )

        top_fig = make_top_view_heatmap(
            df,
            radius_mm=radius_mm,
            cmap=cmap,
            grid_res=grid_res,
            show_labels=show_labels,
        )
        st.plotly_chart(top_fig, use_container_width=True, key="top_view_fig")

    st.markdown("---")
    st.markdown("### 3D 视图")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        cam_box = st.expander("视角（可选）", expanded=False)
        with cam_box:
            eye_cols = st.columns(3)
            eye_x = eye_cols[0].slider("eye.x", -3.0, 3.0, 1.7, 0.1, key="eye_x")
            eye_y = eye_cols[1].slider("eye.y", -3.0, 3.0, 1.7, 0.1, key="eye_y")
            eye_z = eye_cols[2].slider("eye.z", 0.1, 5.0, 1.2, 0.1, key="eye_z")
            camera = (eye_x, eye_y, eye_z)

        fig3d = make_3d_surface(
            df,
            radius_mm=radius_mm,
            cmap=cmap,
            grid_res=grid_res,
            spec_upper=spec_upper,
            spec_lower=spec_lower,
            camera_eye=camera,
        )
        st.plotly_chart(fig3d, use_container_width=True, key="surface3d")


# ============================================================
# Main
# ============================================================

def main():
    st.title("晶圆厚度可视化")

    tab_normal, tab_gen = st.tabs(["📊 普通模式 (Analysis)", "🛠️ 坐标生成器 (Generator)"])

    with tab_normal:
        normal_mode_ui()

    with tab_gen:
        blind_mode_ui()


if __name__ == "__main__":
    main()
