#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NASDAQ Market Decision Dashboard
- 나스닥 종합지수 기준 계산 / 나스닥100·선물은 참고용
- 표준편차(1σ~6σ) 밴드, ATR, 변동성
- 테일러 급수 기반 국소 곡률 분석(실험적)
- GJR-GARCH(1,1,1) 조건부 변동성(레버리지 효과 포함)
- 벨만 최적화(동적계획법) 기반 다단계 포지션 계획
- 0/25/50/75/100% 포지션 레벨
- Streamlit 웹 대시보드

실행:
    pip install -r requirements.txt
    pip install arch   # GJR-GARCH 모형에 필요 (requirements.txt에 없다면 별도 설치)
    streamlit run nasdaq_model.py

데이터:
    yfinance 무료 데이터 사용.
    미국 시장은 야후 파이낸스에 나스닥 선물(NQ=F) 데이터가 비교적 안정적으로 제공되나,
    여전히 방향 점수 등 모든 계산은 나스닥 종합지수(현물)만 사용하도록 설계되어 있다.
    필요하면 사이드바에서 직접 티커를 수정할 수 있다.
"""

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

st.set_page_config(
    page_title="NASDAQ Market Engine",
    page_icon="📈",
    layout="wide",
)

DEFAULTS = {
    "KOSPI 현물": "^IXIC",   # 나스닥 종합지수 (계산 기준)
    "KOSPI200 현물": "^NDX",  # 나스닥100 (참고용)
    "KOSPI200 선물": "NQ=F",  # E-mini 나스닥 선물. 미국시장은 야후에 실제 데이터가 있어 기본값 채움(참고용)
    "USD/KRW": "^TNX",       # 미국 10년물 국채금리 (매크로게이트용, KRW 대체)
    "VIX": "^VIX",
}

# ---- 신뢰도 보정 임계값 (필요시 조정) ----
VOLUME_RATIO_STRONG = 1.5      # 이 배수 이상 거래량이 터져야 '진짜 이탈' 가능성 높음
VOLATILITY_RATIO_HOT = 1.2     # 단기/장기 변동성 비율이 이 값을 넘으면 '과열' 구간
WEEKLY_CONFLUENCE_PCT = 0.005  # 주봉 MA20과 이 비율(0.5%) 이내로 겹치면 '신뢰 구간'
VIX_RISK_LEVEL = 25.0          # VIX 공포지수 위험 기준
KRW_RISK_5D_PCT = 2.0          # 5일간 미 10년물 금리 급등(위험자산 회피) 위험 기준(%, 상대변화율)

# ---- 벨만 최적화(동적계획법) 설정 ----
POSITION_STATES = [0, 25, 50, 75, 100]  # 이산화된 헤지/인버스 포지션 레벨
BELLMAN_HORIZON = 5                      # 몇 영업일 앞까지 다단계로 계획할지

@st.cache_data(ttl=300)
def load_data(ticker: str, period: str = "1y", interval: str = "1d"):
    if not ticker:
        return pd.DataFrame()

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        df = df[required + ([c for c in ["Volume"] if c in df.columns])]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def pct_change(df, n=1):
    return df["Close"].pct_change(n) * 100


def atr(df, n=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def zscore(series, n=60):
    mean = series.rolling(n).mean()
    std = series.rolling(n).std()
    return (series - mean) / std.replace(0, np.nan)


def make_levels(df):
    close = float(df["Close"].iloc[-1])
    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma60 = float(df["Close"].rolling(60).mean().iloc[-1])
    a = float(atr(df).iloc[-1])

    # 변동성에 따른 동적 밴드
    resistance = max(ma20, close) + 1.0 * a
    support = min(ma20, close) - 1.0 * a

    # 최근 60일 고저점
    high60 = float(df["High"].rolling(60).max().iloc[-1])
    low60 = float(df["Low"].rolling(60).min().iloc[-1])

    return {
        "close": close,
        "ma20": ma20,
        "ma60": ma60,
        "atr": a,
        "resistance": resistance,
        "support": support,
        "high60": high60,
        "low60": low60,
    }


def std_bands(df, ma_n=20, std_n=20, max_k=6):
    """
    이동평균(ma_n일) ± k * 표준편차(std_n일) 밴드를 k=1..max_k 까지 계산.
    반환: (기준 이동평균, 표준편차, {k: {"support":..., "resistance":...}})
    """
    close = df["Close"]
    ma = float(close.rolling(ma_n).mean().iloc[-1])
    std = float(close.rolling(std_n).std().iloc[-1])

    bands = {}
    for k in range(1, max_k + 1):
        bands[k] = {
            "support": ma - k * std,
            "resistance": ma + k * std,
        }
    return ma, std, bands


def volume_ratio(df, n=20):
    """당일 거래량 / 최근 n일 평균 거래량. Volume 데이터가 없으면 NaN."""
    if "Volume" not in df.columns or df["Volume"].isna().all():
        return np.nan
    avg_vol = df["Volume"].rolling(n).mean().iloc[-1]
    today_vol = df["Volume"].iloc[-1]
    if not avg_vol or np.isnan(avg_vol) or avg_vol == 0:
        return np.nan
    return float(today_vol / avg_vol)


def volatility_ratio(df, short_n=5, long_n=60):
    """단기(short_n일) 표준편차 / 장기(long_n일) 표준편차. 1보다 크면 변동성 확대 국면."""
    close = df["Close"]
    std_short = close.rolling(short_n).std().iloc[-1]
    std_long = close.rolling(long_n).std().iloc[-1]
    if not std_long or np.isnan(std_long) or std_long == 0:
        return np.nan
    return float(std_short / std_long)


def weekly_ma20(df, ma_n=20):
    """일봉 데이터를 주봉으로 리샘플링한 뒤 ma_n주 이동평균의 마지막 값."""
    weekly_close = df["Close"].resample("W").last().dropna()
    if len(weekly_close) < ma_n:
        return np.nan
    return float(weekly_close.rolling(ma_n).mean().iloc[-1])


def band_reliability_tags(k, level_value, vol_ratio_val, vola_ratio_val, weekly_ma_val):
    """
    특정 σ 밴드 레벨(level_value, k차수)에 대한 신뢰도 태그 목록을 반환.
    - 변동성 과열 국면에서는 1~2σ를 낮은 신뢰도로 표시
    - 거래량이 평소 대비 충분히 터지지 않으면 낮은 신뢰도로 표시
    - 주봉 MA20과 겹치면 '신뢰 구간'으로 가점 표시
    """
    tags = []

    if not np.isnan(vola_ratio_val) and vola_ratio_val > VOLATILITY_RATIO_HOT and k <= 2:
        tags.append("변동성 과열 · 낮은 신뢰")

    if not np.isnan(vol_ratio_val) and vol_ratio_val < VOLUME_RATIO_STRONG:
        tags.append("거래량 부족 · 가짜 이탈 주의")

    if (
        not np.isnan(weekly_ma_val)
        and weekly_ma_val != 0
        and abs(level_value - weekly_ma_val) / weekly_ma_val <= WEEKLY_CONFLUENCE_PCT
    ):
        tags.append("★ 주봉 MA20 겹침 (신뢰 구간)")

    if not tags:
        tags.append("보통")

    return " / ".join(tags)


def macro_gate(krw_df, vix_df):
    """
    미 10년물 국채금리(5일 변화율)와 VIX 수준을 이용한 매크로 위험 게이트.
    반환: (yield_5d_pct, vix_last, is_risk_on, multiplier)
    위험(금리 급등 + VIX 급등) 동시 충족 시 포지션 진입 강도를 절반으로 낮춘다.
    (금리 급등은 기술주 밸류에이션 할인율 상승으로 나스닥에는 특히 부담 요인)
    """
    krw_5d_pct = np.nan
    vix_last = np.nan

    if not krw_df.empty and len(krw_df) > 5:
        krw_5d_pct = float(krw_df["Close"].pct_change(5).iloc[-1] * 100)

    if not vix_df.empty:
        vix_last = float(vix_df["Close"].iloc[-1])

    is_risk = (
        not np.isnan(krw_5d_pct)
        and not np.isnan(vix_last)
        and krw_5d_pct > KRW_RISK_5D_PCT
        and vix_last > VIX_RISK_LEVEL
    )
    multiplier = 0.5 if is_risk else 1.0
    return krw_5d_pct, vix_last, is_risk, multiplier


def taylor_fit(df, window=20, degree=3):
    """
    최근 window일 종가에 degree차 다항식을 피팅하고, 마지막 시점(a=오늘)에서의
    함수값/1차미분(모멘텀)/2차미분(곡률)/3차미분(곡률 변화율)을 계산.
    ※ 실제 주가는 매끄러운 함수가 아니므로 이는 엄밀한 테일러 급수가 아니라
    '국소 다항식 피팅 기반 근사'다. 예측이라기보다 현재 추세의 휘어짐을
    수치화하는 보조 지표로 사용한다.
    """
    close = df["Close"].tail(window).values
    if len(close) < window:
        return None

    x = np.arange(window, dtype=float)
    coeffs = np.polyfit(x, close, degree)
    poly = np.poly1d(coeffs)
    d1 = poly.deriv(1)
    d2 = poly.deriv(2)
    d3 = poly.deriv(3) if degree >= 3 else np.poly1d([0.0])

    a = window - 1  # 기준점 a = 오늘(윈도우 마지막 날)
    fitted = poly(x)
    resid = close - fitted
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    return {
        "f_a": float(poly(a)),
        "f1_a": float(d1(a)),
        "f2_a": float(d2(a)),
        "f3_a": float(d3(a)),
        "rmse": rmse,
        "window": window,
        "degree": degree,
    }


def taylor_projection(fit, horizons=(1, 2, 3, 4, 5)):
    """
    테일러 전개식 f(a+h) ≈ f(a) + f'(a)h + f''(a)/2 h² + f'''(a)/6 h³ 로
    h영업일 뒤 경로를 근사하고, 피팅 잔차(RMSE)를 기준점에서 멀어질수록
    커지는 불확실성 구간(오차 밴드)으로 함께 제시한다.
    """
    f_a, f1, f2, f3, rmse = (
        fit["f_a"], fit["f1_a"], fit["f2_a"], fit["f3_a"], fit["rmse"]
    )
    rows = []
    for h in horizons:
        proj = f_a + f1 * h + (f2 / 2) * h ** 2 + (f3 / 6) * h ** 3
        band = rmse * np.sqrt(1 + h)  # 기준점에서 멀수록 오차 확대(단순 근사)
        rows.append({"h": h, "proj": proj, "upper": proj + band, "lower": proj - band})
    return rows


def turning_point_signal(df, short_window=10, long_window=20, degree=2):
    """
    단기(short_window)와 장기(long_window) 윈도우로 각각 다항식을 피팅해
    2차미분(곡률) 부호를 비교. 부호가 서로 다르면 '변곡점이 임박했을 가능성'으로 본다.
    """
    fit_short = taylor_fit(df, window=short_window, degree=degree)
    fit_long = taylor_fit(df, window=long_window, degree=degree)
    if fit_short is None or fit_long is None:
        return None

    short_curv = fit_short["f2_a"]
    long_curv = fit_long["f2_a"]
    sign_flip = (
        abs(short_curv) > 1e-9
        and abs(long_curv) > 1e-9
        and (short_curv > 0) != (long_curv > 0)
    )
    return {"short_curv": short_curv, "long_curv": long_curv, "sign_flip": sign_flip}


def fit_gjr_garch(kospi_df, min_obs=100):
    """
    로그수익률(%) 기준 GJR-GARCH(1,1,1) 모형 적합.
    arch 패키지의 vol='GARCH', o=1 옵션이 곧 GJR-GARCH(비대칭 지시함수 포함) 사양이다.
    반환: (모형결과 또는 None, 오늘의 조건부 일간변동성(%), 향후 BELLMAN_HORIZON일 변동성 예측 경로(%, ndarray))
    """
    if not ARCH_AVAILABLE:
        return None, np.nan, None

    close = kospi_df["Close"]
    rets = 100 * np.log(close / close.shift(1)).dropna()
    if len(rets) < min_obs:
        return None, np.nan, None

    try:
        am = arch_model(rets, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
        res = am.fit(disp="off")
    except Exception:
        return None, np.nan, None

    today_vol = float(res.conditional_volatility.iloc[-1])  # 일간 변동성(%)

    try:
        fc = res.forecast(horizon=BELLMAN_HORIZON, reindex=False)
        var_path = fc.variance.values[-1]
        vol_path = np.sqrt(var_path)  # 일간 변동성(%) 경로
    except Exception:
        vol_path = np.full(BELLMAN_HORIZON, today_vol)

    return res, today_vol, vol_path


def extended_garch_variance(garch_today_vol_pct, vol_ratio_val, basis_z_val, delta1=0.3, delta2=0.2):
    """
    확장형 조건부분산 h_t_ext = h_t * (1 + δ1·X1 + δ2·X2) 근사.
    원 논문식 h_t = ω + αε² + γε²I(ε<0) + βh_{t-1} + Σδ_j X_{t-j} 를
    MLE로 동시추정하려면 arch 패키지의 우도함수를 직접 뜯어고쳐야 하므로,
    실무적으로는 이미 적합된 GJR-GARCH 분산(h_t)에 외생변수 보정을
    사후적으로(post-hoc) 곱연산으로 얹는 근사 방식을 쓴다.

    X1(거래량 소진): 거래량비율이 1보다 작을수록(평균 이하 거래) 소진 신호로 간주해 0~1로 스케일.
    X2(프리미엄/괴리 극단도): 현·선물 Basis Z-score의 절댓값 (선물 데이터 없으면 0으로 처리됨).
    반환: (h_t 원본, h_t_ext 확장분산) — 둘 다 %^2 단위.
    """
    if np.isnan(garch_today_vol_pct):
        return np.nan, np.nan

    h_t = garch_today_vol_pct ** 2

    x1 = min(max(0.0, 1.0 - vol_ratio_val), 1.0) if not np.isnan(vol_ratio_val) else 0.0
    x2 = abs(basis_z_val) if not np.isnan(basis_z_val) else 0.0

    h_t_ext = h_t * (1.0 + delta1 * x1 + delta2 * x2)
    return h_t, h_t_ext


def fit_garch_m(kospi_df, garch_res):
    """
    GARCH-M(평균결합) 근사: r_t = mu + lambda*sqrt(h_t) + e_t 를
    이미 적합된 GJR-GARCH의 조건부변동성(sqrt(h_t))을 설명변수로 하는
    2단계 OLS로 추정한다. (완전결합 MLE 대비 효율성은 낮지만 방향성 파악엔 충분)
    반환: (mu_hat(%), lambda_hat) — 둘 다 % 수익률 단위 기준.
    """
    if garch_res is None:
        return np.nan, np.nan

    close = kospi_df["Close"]
    rets = 100 * np.log(close / close.shift(1)).dropna()
    cond_vol = garch_res.conditional_volatility

    n = min(len(rets), len(cond_vol))
    if n < 30:
        return np.nan, np.nan

    r = rets.iloc[-n:].values
    h_sqrt = cond_vol.iloc[-n:].values
    X = np.column_stack([np.ones(n), h_sqrt])

    try:
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        mu_hat, lambda_hat = float(beta[0]), float(beta[1])
    except Exception:
        return np.nan, np.nan

    return mu_hat, lambda_hat


def norm_cdf(x):
    """표준정규분포 누적분포함수 (scipy 의존성 없이 math.erf로 구현)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def unhedged_loss_table(mu_daily, sigma_daily_path, spot, z95=1.645):
    """
    '만약 헤지 없이 현물을 그대로 매수/보유했다면'을 가정한 h일 기대손실률 테이블.
    일간 수익률이 매일 독립적으로 N(mu_daily, sigma_t^2)를 따른다는 단순화된
    정규분포 가정 하에, h일 누적 기대수익률/하락확률/95% VaR를 계산한다.
    ※ 실제 수익률은 정규분포가 아니고(두꺼운 꼬리), 매일 독립도 아니므로
    이는 근사적 참고치이지 정밀한 리스크 측정치가 아니다.
    반환: h별 (기대누적수익률, 누적변동성, 하락확률, 95% VaR) 리스트
    """
    rows = []
    cum_var = 0.0
    for h, sigma_t in enumerate(sigma_daily_path, start=1):
        cum_var += sigma_t ** 2
        sigma_cum = math.sqrt(cum_var)
        exp_ret = mu_daily * h
        prob_loss = norm_cdf(-exp_ret / sigma_cum) if sigma_cum > 0 else np.nan
        var95 = exp_ret - z95 * sigma_cum
        rows.append({
            "h": h,
            "exp_ret": exp_ret,
            "sigma_cum": sigma_cum,
            "prob_loss": prob_loss,
            "var95": var95,
        })
    return rows


def hybrid_bands(ma_t, daily_vol_ext_price, k1, k2, gamma, neg_shock_indicator, theta_call_price):
    """
    3번 공식(최종 하이브리드 밴드) 구현:
      Upper_t = MA_t + k1·√h_t_ext - θ_call
      Lower_t = MA_t - (k2 + γ·I(ε<0))·√h_t_ext
    daily_vol_ext_price = √h_t_ext 를 가격 단위로 환산한 값(스팟가격 × %변동성/100).
    gamma는 GJR-GARCH의 비대칭계수(음수 충격 시 추가 증폭), neg_shock_indicator는
    직전 충격(어제 수익률)이 음수였는지(0 또는 1).
    """
    gamma_eff = max(gamma, 0.0) if not np.isnan(gamma) else 0.0
    upper = ma_t + k1 * daily_vol_ext_price - theta_call_price
    lower = ma_t - (k2 + gamma_eff * neg_shock_indicator) * daily_vol_ext_price
    return upper, lower


def bellman_optimal_path(
    current_position,
    mu_daily,
    sigma_daily_path,
    states=POSITION_STATES,
    risk_aversion=3.0,
    rebal_cost=0.02,
):
    """
    벨만의 최적성 원리(backward induction, 가치반복)를 이용해
    향후 len(sigma_daily_path)영업일에 걸친 최적 포지션 경로를 계산한다.

    상태(state) = 헤지/인버스 포지션 비중 (0/25/50/75/100%)
    헤지/인버스 포지션은 기초자산(나스닥)과 반대 방향으로 수익이 나므로,
    기대수익 항에는 -mu_daily(부호 반전)를 사용한다 — 즉 하락 신호(mu<0)일수록
    헤지 비중을 높이는 것이 보상을 극대화한다.

    각 시점의 보상(reward) = 기대수익 - 리스크비용 - 리밸런싱비용
        R(p, p_prev, t) = (p/100)*(-mu_daily)
                          - risk_aversion * (p/100)^2 * sigma_daily_path[t]^2
                          - rebal_cost * |p - p_prev| / 100
    종료조건: V_H(p) = 0 (마지막 시점 이후 가치는 0으로 정규화)
    이 값을 뒤에서부터(backward) 채워나가는 것이 벨만 방정식의 핵심이다:
        V_t(p_prev) = max_p [ R(p, p_prev, t) + V_{t+1}(p) ]

    ※ mu_daily, sigma_daily_path는 실제 확률분포가 아니라 방향점수/GARCH변동성에서
    유도한 단순화된 추정치이므로, 이 경로는 '참고용 다단계 계획'이지 확정적 예측이 아니다.
    """
    hedge_mu = -mu_daily  # 헤지/인버스 포지션 수익은 기초자산과 반대 부호
    H = len(sigma_daily_path)
    V_next = {p: 0.0 for p in states}
    policies = []  # policies[t][p_prev] = 그 시점에서의 최적 다음 포지션

    for t in reversed(range(H)):
        sigma_t = sigma_daily_path[t]
        V_curr = {}
        policy_t = {}
        for p_prev in states:
            best_val, best_p = -np.inf, p_prev
            for p in states:
                reward = (
                    (p / 100.0) * hedge_mu
                    - risk_aversion * ((p / 100.0) ** 2) * (sigma_t ** 2)
                    - rebal_cost * abs(p - p_prev) / 100.0
                )
                val = reward + V_next[p]
                if val > best_val:
                    best_val, best_p = val, p
            V_curr[p_prev] = best_val
            policy_t[p_prev] = best_p
        policies.insert(0, policy_t)
        V_next = V_curr

    v0 = V_next[current_position]

    # 초기 포지션에서 시작해 정책을 따라 앞으로(forward) 시뮬레이션
    path = [current_position]
    p_prev = current_position
    for t in range(H):
        p_next = policies[t][p_prev]
        path.append(p_next)
        p_prev = p_next

    return v0, path


def directional_score(df):
    """-100 ~ +100. +는 상승, -는 하락."""
    if len(df) < 70:
        return 0.0, 0.0, "데이터 부족"

    close = df["Close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    r5 = close.pct_change(5).iloc[-1] * 100
    r20 = close.pct_change(20).iloc[-1] * 100

    score = 0.0

    # 추세
    # 완전히 동일한 값(데이터 오류/거래정지 등 극단적 엣지케이스)일 때 -30/-25로
    # 편향되지 않도록, '차이가 사실상 없는' 경우는 중립(0)으로 처리한다.
    price_eps = close.iloc[-1] * 1e-6  # 상대오차 기준 허용오차
    if close.iloc[-1] > ma20.iloc[-1] + price_eps:
        score += 30
    elif close.iloc[-1] < ma20.iloc[-1] - price_eps:
        score += -30
    # else: 사실상 동일 -> 0 (가산 없음)

    if ma20.iloc[-1] > ma60.iloc[-1] + price_eps:
        score += 25
    elif ma20.iloc[-1] < ma60.iloc[-1] - price_eps:
        score += -25
    # else: 사실상 동일 -> 0 (가산 없음)

    # 단기 모멘텀
    score += np.clip(r5 * 8, -20, 20)
    score += np.clip(r20 * 3, -15, 15)

    # 최근 종가 위치
    hi = close.rolling(60).max().iloc[-1]
    lo = close.rolling(60).min().iloc[-1]
    pos = (close.iloc[-1] - lo) / (hi - lo) if hi != lo else 0.5
    score += (pos - 0.5) * 20

    score = float(np.clip(score, -100, 100))
    confidence = float(np.clip(abs(score), 0, 100))

    if score >= 25:
        direction = "상승 우세"
    elif score <= -25:
        direction = "하락 우세"
    else:
        direction = "중립 / 혼조"

    return score, confidence, direction


def position_level(score, confidence, basis_z=0.0):
    """
    0/25/50/75/100 레벨.
    하락 방향일수록 인버스/헤지 비중을 높이는 구조.
    Basis가 비정상적으로 벌어지면 한 단계 보수적으로 낮춘다.
    """
    effective = score

    # 현물 대비 선물의 괴리가 지나치게 크면 확신도 할인
    if abs(basis_z) >= 2:
        effective *= 0.65
    elif abs(basis_z) >= 1.5:
        effective *= 0.8

    if effective <= -65 and confidence >= 65:
        return 100
    if effective <= -45 and confidence >= 45:
        return 75
    if effective <= -25 and confidence >= 25:
        return 50
    if effective < 0:
        return 25
    return 0


def regime_text(score, confidence):
    if confidence < 25:
        return "신호 약함"
    if score <= -60:
        return "강한 하방 위험"
    if score <= -25:
        return "하방 우세"
    if score >= 60:
        return "강한 상승"
    if score >= 25:
        return "상승 우세"
    return "중립"


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ 데이터 설정")

kospi_ticker = st.sidebar.text_input("나스닥 종합지수 (계산 기준)", DEFAULTS["KOSPI 현물"])
ks200_ticker = st.sidebar.text_input("나스닥100 (참고용, 계산에는 미반영)", DEFAULTS["KOSPI200 현물"])
futures_ticker = st.sidebar.text_input("나스닥 선물 (참고용, 계산에는 미반영)", DEFAULTS["KOSPI200 선물"])

st.sidebar.divider()
st.sidebar.caption("매크로 게이트용 데이터")
krw_ticker = st.sidebar.text_input("미 10년물 국채금리", DEFAULTS["USD/KRW"])
vix_ticker = st.sidebar.text_input("VIX 지수", DEFAULTS["VIX"])

st.sidebar.divider()
st.sidebar.caption("벨만 최적화(다단계 포지션 계획) 파라미터")
risk_aversion = st.sidebar.slider("리스크회피계수", 0.5, 10.0, 3.0, 0.5)
rebal_cost_pct = st.sidebar.slider("리밸런싱 비용 (%)", 0.0, 5.0, 2.0, 0.5)
rebal_cost = rebal_cost_pct / 100.0

st.sidebar.divider()
st.sidebar.caption("확장형 GJR-GARCH + 하이브리드 밴드 파라미터")
delta1_volume = st.sidebar.slider("δ1 거래량 소진 가중치", 0.0, 1.0, 0.3, 0.05)
delta2_basis = st.sidebar.slider("δ2 괴리(Basis) 극단도 가중치", 0.0, 1.0, 0.2, 0.05)
k1_upper = st.sidebar.slider("k1 상단 승수", 0.5, 6.0, 2.0, 0.5)
k2_lower = st.sidebar.slider("k2 하단 승수", 0.5, 6.0, 2.0, 0.5)
theta_call_pct = st.sidebar.slider("θ_call 상단 압축계수 (스팟가격 대비 %)", 0.0, 3.0, 0.0, 0.1)

period = st.sidebar.selectbox(
    "분석 기간",
    ["6mo", "1y", "2y", "5y"],
    index=1,
)

st.sidebar.caption(
    "※ 무료 데이터 제공처의 티커/지연 여부는 변할 수 있습니다. "
    "실전 매매 전에는 거래소/증권사 데이터로 교차검증하세요. "
    "방향 점수/지지·저항선 등 모든 계산은 나스닥 종합지수만 사용하며, "
    "나스닥100과 선물은 티커를 입력해도 화면에 참고용으로만 표시됩니다."
)

# ---------------- Load ----------------
kospi = load_data(kospi_ticker, period)
ks200 = load_data(ks200_ticker, period)
futures = load_data(futures_ticker, period)
krw = load_data(krw_ticker, period)
vix = load_data(vix_ticker, period)

st.title("📊 NASDAQ Market Decision Engine")
st.caption("나스닥 종합지수 기준 계산 · 나스닥100/선물은 참고용 표시")

if kospi.empty:
    st.error(
        "나스닥 종합지수 데이터를 불러오지 못했습니다. "
        "사이드바의 티커(기본값 ^KS11)를 확인하세요."
    )
    st.stop()

# ---------------- Main model ----------------
# 모든 계산(방향 점수/신뢰도/지지·저항 밴드/차트)은 나스닥 종합지수(kospi_ticker, 기본 ^IXIC) 기준.
spot_levels = make_levels(kospi)
spot_score, spot_conf, spot_direction = directional_score(kospi)
spot = spot_levels["close"]

# 나스닥100 현물은 참고용으로만 표시 (계산에는 미반영)
ks200_price = float(ks200["Close"].iloc[-1]) if not ks200.empty else np.nan

fut_score = fut_conf = 0.0
fut_direction = "데이터 없음"
if not futures.empty:
    fut_score, fut_conf, fut_direction = directional_score(futures)

future = float(futures["Close"].iloc[-1]) if not futures.empty else np.nan

# 선물-나스닥100 괴리(Basis)는 내부적으로만 계산하며 화면에는 참고 수치로만 노출한다.
basis = np.nan
basis_z = 0.0
if not np.isnan(future) and not np.isnan(ks200_price) and ks200_price != 0:
    basis = (future - ks200_price) / ks200_price * 100

    if not ks200.empty:
        basis_series = (
            futures["Close"].reindex(ks200.index).ffill() - ks200["Close"]
        ) / ks200["Close"] * 100
        basis_z_series = zscore(basis_series, 60).dropna()
        if not basis_z_series.empty:
            basis_z = float(basis_z_series.iloc[-1])

# 방향 점수/신뢰도는 항상 나스닥 종합지수만으로 계산한다.
# 나스닥100/선물 데이터는 있어도 점수 계산에는 반영하지 않고, 화면에는 참고용으로만 표시한다.
composite_score = spot_score
composite_conf = spot_conf

regime = regime_text(composite_score, composite_conf)
level = position_level(composite_score, composite_conf, basis_z)

# ---- 신뢰도 보정 지표 (거래량 / 변동성 / 주봉 교차 / 매크로) ----
vol_ratio_val = volume_ratio(kospi, n=20)
vola_ratio_val = volatility_ratio(kospi, short_n=5, long_n=60)
weekly_ma_val = weekly_ma20(kospi, ma_n=20)
krw_5d_pct, vix_last, macro_risk, macro_multiplier = macro_gate(krw, vix)

# 매크로 위험 신호(美 금리 급등 + VIX 급등) 시 최종 진입 강도를 절반으로 낮춘다.
level_final = round(level * macro_multiplier / 25) * 25

# ---- GJR-GARCH(1,1,1) 조건부 변동성 ----
garch_res, garch_today_vol_pct, garch_vol_path_pct = fit_gjr_garch(kospi)

# ---- 확장형 조건부분산 (거래량 소진 + 괴리 극단도 외생변수 반영) ----
h_t_raw, h_t_ext = extended_garch_variance(
    garch_today_vol_pct, vol_ratio_val, basis_z, delta1_volume, delta2_basis
)
ext_factor = (h_t_ext / h_t_raw) if (not np.isnan(h_t_raw) and h_t_raw != 0) else np.nan

# ---- GARCH-M(평균결합) 2단계 근사: 변동성 → 기대수익 피드백 ----
garch_m_mu, garch_m_lambda = fit_garch_m(kospi, garch_res)

# ---------------- Header cards ----------------
c1, c3, c4, c5 = st.columns(4)

c1.metric(
    "나스닥 종합지수",
    f"{spot:,.2f}",
    f"{pct_change(kospi).iloc[-1]:+.2f}%"
)

c3.metric("방향 점수", f"{composite_score:+.1f}")
c4.metric("신뢰도", f"{composite_conf:.0f}%")
c5.metric(
    "헤지/인버스 레벨",
    f"{level_final}%",
    None if level_final == level else f"매크로 보정 전 {level}%"
)

if not futures.empty:
    st.caption(
        f"선물 (참고용): {future:,.2f} ({pct_change(futures).iloc[-1]:+.2f}%) "
        "— 방향 점수 계산에는 반영되지 않습니다."
    )

st.divider()

# ---------------- Decision ----------------
left, right = st.columns([1, 1])

with left:
    st.subheader("🎯 현재 판단")
    st.markdown(f"### {regime}")
    st.write(f"**방향:** {spot_direction}")
    st.write(f"**종합 점수:** `{composite_score:+.1f} / 100`")
    st.write(f"**신뢰도:** `{composite_conf:.0f}%`")
    st.write(f"**권장 헤지/인버스 단계:** `{level}%`")
    if level_final != level:
        st.write(f"**매크로 보정 후 최종 진입 강도:** `{level_final}%` (위험 신호로 절반 축소)")

    if composite_score <= -45:
        st.warning(
            "하방 위험이 높은 구간입니다. "
            "단일 진입보다 분할 대응을 우선하세요."
        )
    elif composite_score >= 45:
        st.success(
            "상승 추세 우세입니다. "
            "과도한 헤지는 줄이고 추세 추종 여부를 검토하세요."
        )
    else:
        st.info("신호가 혼조입니다. 신규 포지션은 보수적으로 접근하세요.")

with right:
    st.subheader("📐 현·선물 괴리 (참고용)")
    st.caption("※ 아래 수치는 참고 정보이며, 위 방향 점수/신뢰도 계산에는 반영되지 않습니다.")
    if not np.isnan(basis):
        st.metric("Basis", f"{basis:+.3f}%")
        st.write(f"Basis Z-score: `{basis_z:+.2f}`")

        if abs(basis_z) >= 2:
            st.error("괴리 극단: 방향 신호의 신뢰도를 할인하는 구간")
        elif abs(basis_z) >= 1.5:
            st.warning("괴리 확대: 추격 진입 주의")
        else:
            st.success("괴리 정상 범위")
    else:
        st.info("선물 데이터를 사용하지 않아 Basis는 계산하지 않습니다. (현물 단독 분석 모드)")

# ---------------- Reliability filters ----------------
st.divider()
st.subheader("🧪 신뢰도 보정 필터")

r1, r2, r3 = st.columns(3)

with r1:
    st.markdown("**① 거래량 필터**")
    if not np.isnan(vol_ratio_val):
        st.metric("거래량 비율 (당일 / 20일 평균)", f"{vol_ratio_val:.2f}x")
        if vol_ratio_val >= VOLUME_RATIO_STRONG:
            st.success(f"평균 대비 {VOLUME_RATIO_STRONG}배 이상 → 밴드 터치 신뢰도 높음")
        else:
            st.warning("거래량 부족 → 밴드 터치 시 '가짜 이탈' 가능성")
    else:
        st.info("거래량 데이터를 사용할 수 없습니다.")

with r2:
    st.markdown("**② 변동성 비율**")
    if not np.isnan(vola_ratio_val):
        st.metric("변동성 비율 (5일σ / 60일σ)", f"{vola_ratio_val:.2f}")
        if vola_ratio_val > VOLATILITY_RATIO_HOT:
            st.warning(
                f"{VOLATILITY_RATIO_HOT} 초과 → 변동성 과열 구간. "
                "1~2σ 밴드는 무시하고 3σ 이상만 신뢰 권장"
            )
        else:
            st.success("변동성 정상 범위 → 모든 σ 밴드 참고 가능")
    else:
        st.info("변동성 비율을 계산할 데이터가 부족합니다.")

with r3:
    st.markdown("**④ 매크로 게이트**")
    krw_txt = f"{krw_5d_pct:+.2f}%" if not np.isnan(krw_5d_pct) else "N/A"
    vix_txt = f"{vix_last:.1f}" if not np.isnan(vix_last) else "N/A"
    st.write(f"美 10년물 금리 5일 변화율: `{krw_txt}`")
    st.write(f"VIX: `{vix_txt}`")
    if macro_risk:
        st.error("위험 신호 감지: 美 금리 급등 + VIX 급등 → 진입 강도 50% 축소 적용됨")
    else:
        st.success("매크로 위험 신호 없음")

st.caption(
    "③ 다중 타임프레임(주봉) 교차 검증 결과는 아래 표준편차 밴드 표의 "
    "'신뢰도' 열에 '★ 주봉 MA20 겹침'으로 표시됩니다."
)

# ---------------- Bands ----------------
st.divider()
st.subheader("📏 나스닥 종합지수 동적 밴드")

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("60일 저점", f"{spot_levels['low60']:,.2f}")
b2.metric("지지선", f"{spot_levels['support']:,.2f}")
b3.metric("현재", f"{spot_levels['close']:,.2f}")
b4.metric("저항선", f"{spot_levels['resistance']:,.2f}")
b5.metric("60일 고점", f"{spot_levels['high60']:,.2f}")

st.write(
    f"MA20 `{spot_levels['ma20']:,.2f}` · "
    f"MA60 `{spot_levels['ma60']:,.2f}` · "
    f"ATR14 `{spot_levels['atr']:,.2f}`"
)

# ---------------- Standard deviation bands (1σ~6σ) ----------------
st.divider()
st.subheader("📐 표준편차 밴드 (1σ ~ 6σ)")
st.caption("MA20 기준 ± k × 20일 표준편차. k값이 클수록 통계적으로 드문(극단적인) 구간입니다.")

std_ma, std_val, bands = std_bands(kospi, ma_n=20, std_n=20, max_k=6)

band_rows = []
for k in range(1, 7):
    support_val = bands[k]["support"]
    resistance_val = bands[k]["resistance"]
    band_rows.append({
        "σ 배수": f"{k}σ",
        "지지선 (하단)": f"{support_val:,.2f}",
        "지지선 신뢰도": band_reliability_tags(k, support_val, vol_ratio_val, vola_ratio_val, weekly_ma_val),
        "저항선 (상단)": f"{resistance_val:,.2f}",
        "저항선 신뢰도": band_reliability_tags(k, resistance_val, vol_ratio_val, vola_ratio_val, weekly_ma_val),
    })

st.dataframe(
    pd.DataFrame(band_rows).set_index("σ 배수"),
    width="stretch",
)
st.caption(
    f"기준 MA20: `{std_ma:,.2f}` · 20일 표준편차: `{std_val:,.2f}` · "
    f"주봉 MA20: `{weekly_ma_val:,.2f}`" if not np.isnan(weekly_ma_val)
    else f"기준 MA20: `{std_ma:,.2f}` · 20일 표준편차: `{std_val:,.2f}` · 주봉 MA20: 데이터 부족"
)

# ---------------- Taylor series (local polynomial) curvature analysis ----------------
st.divider()
st.subheader("📈 테일러 급수 기반 곡률 분석 (실험적)")
st.caption(
    "⚠️ 실제 주가는 매끄러운(무한 미분 가능한) 함수가 아니므로, 이 섹션은 "
    "엄밀한 테일러 급수가 아니라 '최근 20일 다항식 국소 피팅' 기반 근사입니다. "
    "예측값이 아니라 현재 추세의 휘어짐(모멘텀·가속도)을 보조적으로 참고하는 용도로만 사용하세요."
)

taylor_window = 20
taylor_degree = 3
tfit = taylor_fit(kospi, window=taylor_window, degree=taylor_degree)

if tfit is None:
    st.info(f"다항식 피팅에 필요한 최근 {taylor_window}일 데이터가 부족합니다.")
else:
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("f(a) 기준값", f"{tfit['f_a']:,.2f}")
    t2.metric("f'(a) 1차미분(모멘텀)", f"{tfit['f1_a']:+.2f} / 일")
    t3.metric("f''(a) 2차미분(곡률/가속도)", f"{tfit['f2_a']:+.3f}")
    t4.metric("f'''(a) 3차미분(곡률 변화율)", f"{tfit['f3_a']:+.4f}")

    if tfit["f2_a"] > 0:
        st.success("곡률 양(+): 하락 속도 둔화 또는 상승 가속 국면 (바닥권/상승 전환 신호 가능)")
    elif tfit["f2_a"] < 0:
        st.warning("곡률 음(-): 상승 속도 둔화 또는 하락 가속 국면 (고점권/하락 전환 신호 가능)")
    else:
        st.info("곡률 거의 없음: 선형적인 추세 구간")

    st.write("**단기 경로 근사 (h영업일 뒤, 테일러 전개 기반)**")
    proj_rows = taylor_projection(tfit, horizons=(1, 2, 3, 4, 5))
    proj_table = pd.DataFrame([
        {
            "h (영업일 뒤)": r["h"],
            "근사 예상값": f"{r['proj']:,.2f}",
            "오차밴드 상단": f"{r['upper']:,.2f}",
            "오차밴드 하단": f"{r['lower']:,.2f}",
        }
        for r in proj_rows
    ]).set_index("h (영업일 뒤)")
    st.dataframe(proj_table, width="stretch")
    st.caption(
        "오차밴드는 최근 20일 피팅 잔차(RMSE)를 기준으로 h가 커질수록(기준점 a에서 "
        "멀어질수록) 넓어지도록 근사한 값입니다. h=1에 가까울수록 신뢰도가 높습니다."
    )

    tp = turning_point_signal(kospi, short_window=10, long_window=taylor_window, degree=2)
    if tp is not None:
        if tp["sign_flip"]:
            st.error(
                f"⚠️ 변곡점 신호: 단기(10일) 곡률({tp['short_curv']:+.3f})과 "
                f"장기(20일) 곡률({tp['long_curv']:+.3f})의 부호가 반대입니다. "
                "추세 전환이 임박했을 가능성 — 위 σ 밴드 터치 시 특히 주의 깊게 확인하세요."
            )
        else:
            st.success("단기/장기 곡률 방향 일치 — 추세 전환 신호 없음")

# ---------------- GJR-GARCH conditional volatility ----------------
st.divider()
st.subheader("📉 GJR-GARCH(1,1,1) 조건부 변동성 모형")

if not ARCH_AVAILABLE:
    st.warning(
        "`arch` 패키지가 설치되어 있지 않습니다. 터미널에서 "
        "`pip install arch` 실행 후 앱을 다시 시작하세요."
    )
elif garch_res is None:
    st.info("GJR-GARCH 적합에 필요한 데이터(최소 100일 이상의 일간수익률)가 부족합니다.")
else:
    params = garch_res.params
    omega = float(params.get("omega", np.nan))
    alpha1 = float(params.get("alpha[1]", np.nan))
    gamma1 = float(params.get("gamma[1]", np.nan))
    beta1 = float(params.get("beta[1]", np.nan))

    g1, g2, g3 = st.columns(3)
    g1.metric("오늘 조건부 일간변동성", f"{garch_today_vol_pct:.3f}%")
    g2.metric("연율화 변동성(참고)", f"{garch_today_vol_pct * np.sqrt(252):.2f}%")
    g3.metric("비대칭계수 γ (레버리지 효과)", f"{gamma1:+.4f}" if not np.isnan(gamma1) else "N/A")

    if not np.isnan(gamma1) and gamma1 > 0:
        st.error(
            f"γ = {gamma1:+.4f} > 0 → 레버리지 효과 확인: 음(-)의 충격(하락)이 "
            "양(+)의 충격(상승)보다 변동성을 더 크게 증폭시키는 구조입니다."
        )
    elif not np.isnan(gamma1):
        st.info(f"γ = {gamma1:+.4f} ≤ 0 → 이 구간에서는 뚜렷한 레버리지 효과가 관측되지 않습니다.")

    st.caption(
        f"ω(상수항)=`{omega:.4f}` · α(충격계수)=`{alpha1:.4f}` · "
        f"γ(비대칭계수)=`{gamma1:.4f}` · β(지속성계수)=`{beta1:.4f}` "
        "(모두 % 수익률 기준 GJR-GARCH(1,1,1) 적합 파라미터)"
    )

    st.write(f"**향후 {BELLMAN_HORIZON}영업일 조건부 변동성 예측 경로**")
    garch_rows = []
    for h, vol_pct in enumerate(garch_vol_path_pct, start=1):
        band_1s = spot * (vol_pct / 100.0)
        garch_rows.append({
            "h (영업일 뒤)": h,
            "예측 일간변동성(%)": f"{vol_pct:.3f}%",
            "±1σ 가격폭(참고)": f"±{band_1s:,.2f}",
        })
    st.dataframe(pd.DataFrame(garch_rows).set_index("h (영업일 뒤)"), width="stretch")
    st.caption(
        "이 GARCH 변동성은 아래 '확장형 분산' 및 '벨만 최적화' 모듈의 "
        "리스크(위험) 항으로 이어져 사용됩니다."
    )

    # ---------------- Extended variance + GARCH-M + Hybrid bands ----------------
    st.divider()
    st.subheader("🧬 확장형 GJR-GARCH + GARCH-M + 하이브리드 밴드")
    st.caption(
        "⚠️ 실제 미결제약정/옵션프리미엄 데이터는 무료로 구할 수 없어, "
        "거래량비율(δ1)과 현·선물 괴리 Z-score(δ2)를 파생 수급 소진의 대리변수로 사용합니다. "
        "또한 arch 패키지가 분산방정식에 외생변수를 직접 MLE로 넣는 것을 지원하지 않으므로, "
        "이미 적합된 GJR-GARCH 분산에 사후 보정을 가하는 근사 방식입니다."
    )

    e1, e2, e3 = st.columns(3)
    e1.metric("h_t (원 GJR-GARCH 분산)", f"{h_t_raw:.4f} %²" if not np.isnan(h_t_raw) else "N/A")
    e2.metric("h_t,ext (외생변수 반영 후)", f"{h_t_ext:.4f} %²" if not np.isnan(h_t_ext) else "N/A")
    e3.metric("확장 배수", f"{ext_factor:.2f}x" if not np.isnan(ext_factor) else "N/A")

    if not np.isnan(ext_factor) and ext_factor > 1.3:
        st.warning("거래량 소진 또는 괴리 극단도가 높아 실질 리스크가 원 GARCH 추정치보다 크게 확대되었습니다.")

    st.write("**② GARCH-M 피드백 (변동성 → 기대수익 연동, 2단계 근사)**")
    gm1, gm2 = st.columns(2)
    gm1.metric("μ̂ (상수항, %)", f"{garch_m_mu:+.4f}%" if not np.isnan(garch_m_mu) else "N/A")
    gm2.metric("λ̂ (위험프리미엄 계수)", f"{garch_m_lambda:+.4f}" if not np.isnan(garch_m_lambda) else "N/A")
    if not np.isnan(garch_m_lambda):
        if garch_m_lambda > 0:
            st.success("λ̂ > 0: 변동성이 커질수록 반등 기대수익도 함께 커지는 구조(변동성-수익 양의 피드백)")
        else:
            st.info("λ̂ ≤ 0: 이 구간에서는 변동성 확대가 기대수익 상승으로 이어지지 않습니다.")

    st.write("**③ 최종 하이브리드 밴드**")
    daily_vol_ext_price = spot * (np.sqrt(h_t_ext) / 100.0) if not np.isnan(h_t_ext) else np.nan
    last_ret = float(kospi["Close"].pct_change().iloc[-1])
    neg_shock_indicator = 1 if last_ret < 0 else 0
    theta_call_price = spot * (theta_call_pct / 100.0)

    if not np.isnan(daily_vol_ext_price):
        hybrid_upper, hybrid_lower = hybrid_bands(
            ma_t=spot_levels["ma20"],
            daily_vol_ext_price=daily_vol_ext_price,
            k1=k1_upper,
            k2=k2_lower,
            gamma=gamma1,
            neg_shock_indicator=neg_shock_indicator,
            theta_call_price=theta_call_price,
        )
        hb1, hb2 = st.columns(2)
        hb1.metric("Upper Band (상단, 압축)", f"{hybrid_upper:,.2f}")
        hb2.metric("Lower Band (하단, 확장)", f"{hybrid_lower:,.2f}")
        st.caption(
            f"어제 충격 부호 I(ε<0) = {neg_shock_indicator} · γ(비대칭계수) = {gamma1:+.4f} · "
            f"θ_call = {theta_call_price:,.2f} · k1 = {k1_upper} · k2 = {k2_lower}. "
            "음의 충격이 있었던 날 다음에는 하단이 자동으로 더 넓게 열립니다."
        )
    else:
        st.info("확장분산을 계산할 수 없어 하이브리드 밴드를 표시할 수 없습니다.")

# ---------------- Bellman optimal multi-step position path ----------------
st.divider()
st.subheader("🧮 벨만 최적화 기반 다단계 포지션 계획 (실험적)")
st.caption(
    "벨만의 최적성 원리(동적계획법)로 향후 "
    f"{BELLMAN_HORIZON}영업일에 걸친 최적 포지션 경로를 계산합니다. "
    "기대수익은 방향점수 + GARCH-M 피드백에서, 리스크는 확장형 GJR-GARCH "
    "변동성(거래량 소진·괴리 극단도 반영)에서 가져오며, "
    "포지션을 바꿀 때마다 리밸런싱 비용을 반영합니다. "
    "⚠️ 실제 확률분포가 아닌 단순화된 추정이므로 참고용 계획이지 확정 예측이 아닙니다."
)

# 기대수익률(하루, 소수) 추정: 방향점수 기반 + GARCH-M(변동성→기대수익 피드백) 항 결합
daily_ret_std = float(kospi["Close"].pct_change().dropna().std())
mu_daily_score = (composite_score / 100.0) * daily_ret_std

if not np.isnan(garch_m_lambda) and not np.isnan(h_t_ext):
    mu_daily_garch_m = (garch_m_lambda * np.sqrt(h_t_ext)) / 100.0  # % → 소수
else:
    mu_daily_garch_m = 0.0

mu_daily = mu_daily_score + mu_daily_garch_m

# 리스크(일간 변동성, 소수) 경로: 확장형 GJR-GARCH 분산을 우선 사용, 없으면 표준편차로 대체
if garch_res is not None and garch_vol_path_pct is not None and not np.isnan(ext_factor):
    sigma_daily_path = (garch_vol_path_pct * np.sqrt(ext_factor)) / 100.0
elif garch_res is not None and garch_vol_path_pct is not None:
    sigma_daily_path = (garch_vol_path_pct / 100.0)
else:
    fallback_sigma = std_val / spot if spot else daily_ret_std
    sigma_daily_path = np.full(BELLMAN_HORIZON, fallback_sigma)

v0, optimal_path = bellman_optimal_path(
    current_position=level_final,
    mu_daily=mu_daily,
    sigma_daily_path=sigma_daily_path,
    states=POSITION_STATES,
    risk_aversion=risk_aversion,
    rebal_cost=rebal_cost,
)

bp1, bp2 = st.columns(2)
bp1.metric("현재 포지션 (t=0)", f"{level_final}%")
bp2.metric(
    f"{BELLMAN_HORIZON}일 뒤 권장 포지션",
    f"{optimal_path[-1]}%",
    f"{optimal_path[-1] - level_final:+d}%p"
)

path_table = pd.DataFrame({
    "시점": ["오늘(t=0)"] + [f"t+{i}일" for i in range(1, len(optimal_path))],
    "권장 포지션(%)": optimal_path,
})
st.dataframe(path_table.set_index("시점"), width="stretch")

if optimal_path[1] != level_final:
    direction_word = "확대" if optimal_path[1] > level_final else "축소"
    st.warning(
        f"다음 거래일 최적 행동: 포지션을 {level_final}% → {optimal_path[1]}%로 "
        f"{direction_word}하는 것이 (현재 가정 하에) 기대효용을 극대화합니다."
    )
else:
    st.success("다음 거래일 최적 행동: 현재 포지션 유지")

with st.expander("벨만 모형 가정 상세"):
    st.write(f"- 방향점수 기반 기대수익: `{mu_daily_score * 100:+.4f}%` (방향점수 {composite_score:+.1f} 반영)")
    st.write(f"- GARCH-M 피드백 기대수익: `{mu_daily_garch_m * 100:+.4f}%` (λ̂={garch_m_lambda:+.4f})" if not np.isnan(garch_m_lambda) else "- GARCH-M 피드백: 계산 불가")
    st.write(f"- 합산 기대수익(mu): `{mu_daily * 100:+.4f}%`")
    st.write("- 헤지/인버스 포지션 수익은 부호가 반대이므로, 하락신호(mu<0)일수록 헤지 비중 확대가 유리하게 계산됩니다.")
    st.write(f"- 리스크회피계수: `{risk_aversion}` · 리밸런싱 비용: `{rebal_cost * 100:.1f}%`")
    st.write(f"- 상태공간: `{POSITION_STATES}` · 계획 기간: `{BELLMAN_HORIZON}`영업일")
    st.write(f"- t=0 가치함수 V(현재 포지션 {level_final}%): `{v0:.6f}`")

# ---------------- Unhedged spot expected loss ----------------
st.divider()
st.subheader("📉 (만약) 헤지 없이 현물만 매수·보유했다면 — 기대손실률")
st.caption(
    "위 헤지 로직을 전혀 모르고 나스닥 현물에 그대로 노출된 경우를 가정한 참고 수치입니다. "
    "일간 수익률이 매일 독립적으로 정규분포를 따른다는 단순화된 가정을 사용하므로, "
    "실제 급락(두꺼운 꼬리 리스크)은 여기 표시된 값보다 더 클 수 있습니다."
)

loss_rows = unhedged_loss_table(mu_daily, sigma_daily_path, spot, z95=1.645)
loss_table = pd.DataFrame([
    {
        "h (영업일 뒤)": r["h"],
        "기대 누적수익률": f"{r['exp_ret']*100:+.2f}%",
        "기대 손익(가격)": f"{spot*r['exp_ret']:+,.1f}",
        "하락 확률": f"{r['prob_loss']*100:.1f}%" if not np.isnan(r["prob_loss"]) else "N/A",
        "95% VaR (손실 하한)": f"{r['var95']*100:+.2f}%",
        "95% VaR (가격)": f"{spot*r['var95']:+,.1f}",
    }
    for r in loss_rows
]).set_index("h (영업일 뒤)")

st.dataframe(loss_table, width="stretch")

worst = loss_rows[-1]
if worst["exp_ret"] < 0:
    st.error(
        f"헤지 없이 {BELLMAN_HORIZON}영업일 보유 시 기대 누적손실률 "
        f"**{worst['exp_ret']*100:+.2f}%**(가격 {spot*worst['exp_ret']:+,.1f}), "
        f"손실 확률 **{worst['prob_loss']*100:.1f}%**, "
        f"95% 신뢰수준 최악의 경우(VaR) **{worst['var95']*100:+.2f}%**"
        f"(가격 {spot*worst['var95']:+,.1f})까지 열려 있습니다. "
        "이게 바로 위 벨만 모형이 헤지 비중을 높게 권고하는 이유입니다."
    )
else:
    st.success(
        f"현재 신호는 상승 우세라 헤지 없이 현물만 보유해도 "
        f"{BELLMAN_HORIZON}영업일 기대수익률이 {worst['exp_ret']*100:+.2f}%로 양(+)입니다."
    )

# ---------------- Chart ----------------
chart_df = kospi[["Close"]].copy()
chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
chart_df["MA60"] = chart_df["Close"].rolling(60).mean()
chart_df = chart_df.tail(180)

st.line_chart(chart_df)

# ---------------- Detailed data ----------------
with st.expander("🔎 상세 분석"):
    rows = {
        "현물 방향 점수": spot_score,
        "현물 신뢰도": spot_conf,
        "선물 방향 점수": fut_score,
        "선물 신뢰도": fut_conf,
        "Basis %": basis,
        "Basis Z": basis_z,
        "권장 헤지/인버스(보정 전)": level,
        "권장 헤지/인버스(매크로 보정 후)": level_final,
        "거래량 비율": vol_ratio_val,
        "변동성 비율 (5일/60일)": vola_ratio_val,
        "주봉 MA20": weekly_ma_val,
        "美 10년물 금리 5일 변화율 %": krw_5d_pct,
        "VIX": vix_last,
        "매크로 위험 신호": macro_risk,
    }
    if tfit is not None:
        rows.update({
            "Taylor f'(a) 모멘텀": tfit["f1_a"],
            "Taylor f''(a) 곡률": tfit["f2_a"],
            "Taylor f'''(a) 곡률변화율": tfit["f3_a"],
            "Taylor 피팅 RMSE": tfit["rmse"],
        })
    if garch_res is not None:
        rows.update({
            "GARCH 오늘 조건부변동성(%)": garch_today_vol_pct,
            "GARCH 비대칭계수 γ": float(garch_res.params.get("gamma[1]", np.nan)),
            "h_t (원 분산)": h_t_raw,
            "h_t,ext (확장 분산)": h_t_ext,
            "확장 배수": ext_factor,
            "GARCH-M λ̂": garch_m_lambda,
            "GARCH-M μ̂(%)": garch_m_mu,
        })
    rows.update({
        "벨만 t=0 가치함수": v0,
        f"벨만 {BELLMAN_HORIZON}일뒤 권장 포지션": optimal_path[-1],
        f"무헤지 {BELLMAN_HORIZON}일 기대손실률(%)": worst["exp_ret"] * 100,
        f"무헤지 {BELLMAN_HORIZON}일 하락확률(%)": worst["prob_loss"] * 100 if not np.isnan(worst["prob_loss"]) else np.nan,
        f"무헤지 {BELLMAN_HORIZON}일 95%VaR(%)": worst["var95"] * 100,
    })
    st.dataframe(
        pd.DataFrame(rows, index=["값"]).T,
        width="stretch",
    )

st.divider()
st.caption(
    "⚠️ 본 프로그램은 투자 판단 보조용입니다. "
    "수익을 보장하지 않으며 실시간 주문/매매 기능은 포함하지 않습니다."
)
