#!/usr/bin/env python3
"""
EXPERIMENT #084 - REGIME_ADAPTIVE_ENSEMBLE_15M_4H_V1
==================================================================================================
Hypothesis: Adaptive regime switching between trend-following and mean-reversion based on 
volatility percentile + trend strength. Use 15m for entries, 4h for trend bias.

Key innovations:
- Dual-regime system: Trend regime (low BBW + strong ADX) vs MR regime (high BBW + weak ADX)
- Three-signal ensemble: HMA trend, RSI momentum, Bollinger position
- Confidence-weighted position sizing: more agreement = larger position
- 4h HMA slope as major trend filter (only trade with higher TF trend)
- Clean array handling to avoid read-only issues

Why this should beat #083 (Sharpe=0.051):
- Simpler, cleaner signal logic without complex nested conditions
- Better regime detection using both BBW and ADX
- Proper 4h trend alignment filter
- Discrete position levels to reduce churn costs

Position sizing:
- MAX signal: 0.35 (3 signals agree in favorable regime)
- MED signal: 0.25 (2 signals agree)
- MIN signal: 0.15 (1 signal only)
- leverage=1.0 (no leverage, risk controlled by position size)
"""

import numpy as np
import pandas as pd

name = "regime_adaptive_ensemble_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period=16):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half, min_periods=half).mean().values
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    diff = 2 * wma1 - wma2
    hma = pd.Series(diff).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
    
    return np.nan_to_num(hma, nan=0)


def calculate_rsi(close, period=14):
    """RSI with proper warmup"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.ones(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / (avg_loss[mask] + 1e-10)
    
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    return rsi


def calculate_atr(high, low, close, period=14):
    """ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return np.nan_to_num(atr, nan=0)


def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)
    
    # Filter DM
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smooth with Wilder's method
    atr_smooth = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / (atr_smooth + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / (atr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return np.nan_to_num(adx, nan=0)


def calculate_bbw(close, period=20):
    """Bollinger Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + 2.0 * std
    lower = sma - 2.0 * std
    bbw = (upper - lower) / (sma + 1e-10)
    
    return np.nan_to_num(bbw, nan=0)


def calculate_bbw_percentile(bbw, lookback=100):
    """BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def resample_to_higher_tf(close, high, low, bars_per_tf):
    """Resample OHLCV to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 2:
        return close[-1:], high[-1:], low[-1:]
    
    c_tf = np.array([close[(i + 1) * bars_per_tf - 1] for i in range(n_tf)])
    h_tf = np.array([np.max(high[i * bars_per_tf:(i + 1) * bars_per_tf]) for i in range(n_tf)])
    l_tf = np.array([np.min(low[i * bars_per_tf:(i + 1) * bars_per_tf]) for i in range(n_tf)])
    
    return c_tf, h_tf, l_tf


def map_tf_to_base(tf_array, bars_per_tf, base_length):
    """Map higher timeframe array back to base timeframe"""
    n_tf = len(tf_array)
    mapped = np.zeros(base_length)
    
    for i in range(base_length):
        tf_idx = min(i // bars_per_tf, n_tf - 1)
        mapped[i] = tf_array[tf_idx]
    
    return mapped


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Position sizing levels
    SIZE_LOW = 0.15
    SIZE_MED = 0.25
    SIZE_HIGH = 0.35
    
    # Regime thresholds
    BBW_TREND_THRESHOLD = 0.45  # Low BBW = trend regime
    BBW_MR_THRESHOLD = 0.70     # High BBW = mean reversion regime
    ADX_TREND_THRESHOLD = 25    # ADX > 25 = strong trend
    
    # RSI thresholds by regime
    RSI_TREND_LONG = 52
    RSI_TREND_SHORT = 48
    RSI_MR_LONG = 35
    RSI_MR_SHORT = 65
    
    # Timeframe: 15m base, 4h trend filter (16 bars per 4h)
    bars_per_4h = 16
    
    # === BASE TIMEFRAME (15m) INDICATORS ===
    hma_15m = calculate_hma(close, period=16)
    hma_fast_15m = calculate_hma(close, period=8)
    rsi_15m = calculate_rsi(close, period=14)
    atr_15m = calculate_atr(high, low, close, period=14)
    adx_15m = calculate_adx(high, low, close, period=14)
    bbw_15m = calculate_bbw(close, period=20)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # === 4H TIMEFRAME INDICATORS (trend filter) ===
    c_4h, h_4h, l_4h = resample_to_higher_tf(close, high, low, bars_per_4h)
    hma_4h = calculate_hma(c_4h, period=16)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    bbw_4h = calculate_bbw(c_4h, period=20)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators to 15m
    hma_4h_mapped = map_tf_to_base(hma_4h, bars_per_4h, n)
    adx_4h_mapped = map_tf_to_base(adx_4h, bars_per_4h, n)
    bbw_pct_4h_mapped = map_tf_to_base(bbw_pct_4h, bars_per_4h, n)
    
    # === REGIME DETECTION ===
    # Trend regime: low volatility + strong trend
    is_trend_regime = (bbw_pct_4h_mapped < BBW_TREND_THRESHOLD) & (adx_4h_mapped > ADX_TREND_THRESHOLD)
    
    # Mean reversion regime: high volatility + weak trend
    is_mr_regime = (bbw_pct_4h_mapped > BBW_MR_THRESHOLD) & (adx_4h_mapped < ADX_TREND_THRESHOLD)
    
    # Neutral regime: everything else
    is_neutral = (~is_trend_regime) & (~is_mr_regime)
    
    # === 4H TREND BIAS ===
    hma_4h_slope = np.zeros(n)
    hma_4h_slope[16:] = np.sign(hma_4h_mapped[16:] - hma_4h_mapped[:-16])
    
    bullish_bias = hma_4h_slope > 0
    bearish_bias = hma_4h_slope < 0
    
    # === SIGNAL 1: HMA TREND ===
    hma_signal = np.zeros(n)
    hma_valid = hma_15m > 0
    
    # Price above HMA + HMA sloping up
    hma_slope = np.zeros(n)
    hma_slope[8:] = np.sign(hma_15m[8:] - hma_15m[:-8])
    
    hma_long = hma_valid & (close > hma_15m) & (hma_slope > 0)
    hma_short = hma_valid & (close < hma_15m) & (hma_slope < 0)
    
    hma_signal = np.where(hma_long, 1, hma_signal)
    hma_signal = np.where(hma_short, -1, hma_signal)
    
    # === SIGNAL 2: RSI MOMENTUM (regime-dependent) ===
    rsi_signal = np.zeros(n)
    
    # Trend regime: RSI confirms direction
    rsi_trend_long = is_trend_regime & (rsi_15m > RSI_TREND_LONG)
    rsi_trend_short = is_trend_regime & (rsi_15m < RSI_TREND_SHORT)
    
    # Mean reversion regime: RSI extremes
    rsi_mr_long = is_mr_regime & (rsi_15m < RSI_MR_LONG)
    rsi_mr_short = is_mr_regime & (rsi_15m > RSI_MR_SHORT)
    
    # Neutral regime: standard RSI
    rsi_neutral_long = is_neutral & (rsi_15m > 55)
    rsi_neutral_short = is_neutral & (rsi_15m < 45)
    
    rsi_signal = np.where(rsi_trend_long | rsi_mr_long | rsi_neutral_long, 1, rsi_signal)
    rsi_signal = np.where(rsi_trend_short | rsi_mr_short | rsi_neutral_short, -1, rsi_signal)
    
    # === SIGNAL 3: BOLLINGER POSITION ===
    bb_signal = np.zeros(n)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    
    # In trend regime: price above middle = long
    bb_trend_long = is_trend_regime & (close > sma_20)
    bb_trend_short = is_trend_regime & (close < sma_20)
    
    # In MR regime: price at extremes = fade
    bb_mr_long = is_mr_regime & (close < bb_lower)
    bb_mr_short = is_mr_regime & (close > bb_upper)
    
    # Neutral: standard BB
    bb_neutral_long = is_neutral & (close < sma_20)
    bb_neutral_short = is_neutral & (close > sma_20)
    
    bb_signal = np.where(bb_trend_long | bb_mr_long | bb_neutral_long, 1, bb_signal)
    bb_signal = np.where(bb_trend_short | bb_mr_short | bb_neutral_short, -1, bb_signal)
    
    # === CONFIDENCE-BASED POSITION SIZING ===
    votes_long = np.zeros(n)
    votes_short = np.zeros(n)
    
    # HMA vote (weighted by 4h bias)
    hma_long_aligned = hma_signal == 1
    hma_short_aligned = hma_signal == -1
    
    # Add full vote if aligned with 4h trend, half vote if not
    votes_long = np.where(hma_long_aligned & bullish_bias, votes_long + 1.0, votes_long)
    votes_long = np.where(hma_long_aligned & (~bullish_bias), votes_long + 0.5, votes_long)
    votes_short = np.where(hma_short_aligned & bearish_bias, votes_short + 1.0, votes_short)
    votes_short = np.where(hma_short_aligned & (~bearish_bias), votes_short + 0.5, votes_short)
    
    # RSI vote
    votes_long = np.where(rsi_signal == 1, votes_long + 1.0, votes_long)
    votes_short = np.where(rsi_signal == -1, votes_short + 1.0, votes_short)
    
    # BB vote
    votes_long = np.where(bb_signal == 1, votes_long + 1.0, votes_long)
    votes_short = np.where(bb_signal == -1, votes_short + 1.0, votes_short)
    
    # === GENERATE FINAL SIGNALS ===
    signals = np.zeros(n)
    
    # Long signals
    long_3sig = (votes_long >= 2.5) & (votes_long > votes_short)
    long_2sig = (votes_long >= 1.5) & (votes_long < 2.5) & (votes_long > votes_short)
    long_1sig = (votes_long >= 0.5) & (votes_long < 1.5) & (votes_long > votes_short)
    
    signals = np.where(long_3sig, SIZE_HIGH, signals)
    signals = np.where(long_2sig, SIZE_MED, signals)
    signals = np.where(long_1sig, SIZE_LOW, signals)
    
    # Short signals
    short_3sig = (votes_short >= 2.5) & (votes_short > votes_long)
    short_2sig = (votes_short >= 1.5) & (votes_short < 2.5) & (votes_short > votes_long)
    short_1sig = (votes_short >= 0.5) & (votes_short < 1.5) & (votes_short > votes_long)
    
    signals = np.where(short_3sig, -SIZE_HIGH, signals)
    signals = np.where(short_2sig, -SIZE_MED, signals)
    signals = np.where(short_1sig, -SIZE_LOW, signals)
    
    # === ATR VOLATILITY FILTER ===
    atr_pct = atr_15m / (close + 1e-10)
    atr_pct = np.nan_to_num(atr_pct, nan=0)
    
    valid_atr = atr_pct > 0
    if np.sum(valid_atr) > 50:
        high_vol_threshold = np.percentile(atr_pct[valid_atr], 95)
        extreme_vol_mask = atr_pct > high_vol_threshold
        signals = np.where(extreme_vol_mask, signals * 0.5, signals)
    
    # === SIGNAL PERSISTENCE (reduce churn) ===
    persistent_signals = np.zeros(n)
    prev_signal = 0.0
    
    first_valid = max(100, 48 * bars_per_4h)
    
    for i in range(n):
        if i < first_valid:
            persistent_signals[i] = 0.0
            prev_signal = 0.0
        elif i < 2:
            persistent_signals[i] = 0.0
            prev_signal = 0.0
        else:
            current = signals[i]
            
            # Same direction as previous - keep it
            if current > 0 and prev_signal > 0:
                persistent_signals[i] = current
                prev_signal = current
            elif current < 0 and prev_signal < 0:
                persistent_signals[i] = current
                prev_signal = current
            # Flipping from flat or opposite - require confirmation
            elif current > 0 and prev_signal <= 0:
                if signals[i-1] > 0:
                    persistent_signals[i] = current
                    prev_signal = current
                else:
                    persistent_signals[i] = prev_signal
            elif current < 0 and prev_signal >= 0:
                if signals[i-1] < 0:
                    persistent_signals[i] = current
                    prev_signal = current
                else:
                    persistent_signals[i] = prev_signal
            # Going to flat
            elif current == 0:
                persistent_signals[i] = 0.0
                prev_signal = 0.0
            else:
                persistent_signals[i] = prev_signal
    
    signals = persistent_signals
    
    # === FINAL CLEANUP ===
    signals = np.nan_to_num(signals, nan=0.0)
    signals = np.clip(signals, -SIZE_HIGH, SIZE_HIGH)
    
    return signals