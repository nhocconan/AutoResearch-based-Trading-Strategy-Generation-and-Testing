#!/usr/bin/env python3
"""
EXPERIMENT #080 - SIMPLIFIED_MTF_REGIME_ENSEMBLE_15M_V4
==================================================================================================
Hypothesis: Simplify the ensemble to 3 core signals (HMA trend, Supertrend, RSI) with clean
regime detection (BBW percentile + ADX). Fix the syntax errors from #079 by removing complex
position management loops and using vectorized signal generation instead.

Key improvements over #079:
- Remove complex position management loops (source of syntax errors)
- Use pure vectorized signal generation (no per-bar state tracking)
- Cleaner regime detection with BBW percentile + ADX threshold
- 3-signal ensemble: HMA slope, Supertrend direction, RSI momentum
- Discrete signal levels (0.0, ±0.25, ±0.35) to minimize churn costs
- 15m entries with 4h trend filter (proven in #070, #072)

Why this should work:
- #070 achieved Sharpe=1.256 with similar MTF approach
- #072 achieved Sharpe=0.589 with HMA+ST+RSI combination
- Simpler code = fewer bugs, faster execution (avoid #078 timeout)
- Regime-aware sizing reduces drawdown in choppy markets

Position sizing:
- MAX signal: 0.35 (controls drawdown during 2022 crash)
- Discrete levels: 0.0, ±0.25, ±0.35
- Regime-based: higher confidence in trend regime, lower in mean-reversion
"""

import numpy as np
import pandas as pd

name = "simplified_mtf_regime_ensemble_15m_v4"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing - fully vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, short_period=16, long_period=48):
    """Calculate Hull Moving Average - vectorized with pandas"""
    n = len(close)
    if n < long_period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    
    def wma(x, period):
        weights = np.arange(1, period + 1)
        return x.rolling(window=period).apply(lambda y: np.sum(y * weights) / np.sum(weights), raw=True)
    
    wma_short = wma(close_series, short_period)
    wma_long = wma(close_series, long_period)
    
    sqrt_long = int(np.sqrt(long_period))
    diff = 2 * wma_short - wma_long
    
    hma = wma(diff, sqrt_long)
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI - vectorized"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.ones(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX - vectorized"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    plus_dm[1:] = np.maximum(0, high[1:] - high[:-1])
    minus_dm[1:] = np.maximum(0, low[:-1] - low[1:])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask = di_sum > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    adx[np.isnan(adx)] = 0
    
    return adx


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    middle = rolling_mean
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    
    bbw = np.zeros(n)
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    bbw[np.isnan(bbw)] = 0
    
    return upper, middle, lower, bbw


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    mid = (high + low) / 2
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i-1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
    
    return supertrend, trend_direction


def resample_to_timeframe(close, high, low, bars_per_tf):
    """Resample data to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return np.array([close[-1]]), np.array([high[-1]]), np.array([low[-1]])
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        c_tf[i] = close[end_idx - 1]
        h_tf[i] = np.max(high[start_idx:end_idx])
        l_tf[i] = np.min(low[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf


def map_tf_to_base(tf_array, bars_per_tf, base_length):
    """Map higher timeframe array back to base timeframe"""
    mapped = np.zeros(base_length)
    n_tf = len(tf_array)
    
    for i in range(base_length):
        tf_idx = min(i // bars_per_tf, n_tf - 1)
        mapped[i] = tf_array[tf_idx]
    
    return mapped


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection - vectorized"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Position sizing constants (MAX 0.35 to control drawdown)
    SIZE_LOW = 0.25
    SIZE_HIGH = 0.35
    
    # Regime thresholds
    ADX_TREND_THRESHOLD = 25
    BBW_TREND_PERCENTILE = 0.40
    BBW_MR_PERCENTILE = 0.70
    
    # RSI thresholds by regime
    RSI_TREND_LONG = 55
    RSI_TREND_SHORT = 45
    RSI_MR_LONG = 35
    RSI_MR_SHORT = 65
    
    # Timeframe conversion: 4h = 16 x 15m
    bars_per_4h = 16
    
    # Base timeframe (15m) indicators
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, short_period=16, long_period=48)
    adx_15m = calculate_adx(high, low, close, period=14)
    bb_upper_15m, bb_mid_15m, bb_lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h timeframe indicators for trend filter
    c_4h, h_4h, l_4h = resample_to_timeframe(close, high, low, bars_per_4h)
    hma_4h = calculate_hma(c_4h, short_period=16, long_period=48)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators to 15m
    hma_4h_mapped = map_tf_to_base(hma_4h, bars_per_4h, n)
    adx_4h_mapped = map_tf_to_base(adx_4h, bars_per_4h, n)
    bbw_pct_4h_mapped = map_tf_to_base(bbw_pct_4h, bars_per_4h, n)
    
    # Calculate HMA slope (trend direction)
    hma_slope_15m = np.zeros(n)
    hma_slope_15m[5:] = np.sign(hma_15m[5:] - hma_15m[:-5])
    
    # Calculate 4h trend
    trend_4h = np.zeros(n)
    valid_4h = hma_4h_mapped > 0
    trend_4h[valid_4h] = np.sign(c_4h[np.arange(n) // bars_per_4h] - hma_4h_mapped[valid_4h])
    trend_4h = np.clip(trend_4h, -1, 1)
    
    # Minimum warmup period
    first_valid = max(200, 100 * bars_per_4h, 48, 45)
    
    # Initialize signals
    signals = np.zeros(n)
    
    # Vectorized regime detection
    is_trend_regime = (adx_4h_mapped > ADX_TREND_THRESHOLD) & (bbw_pct_4h_mapped < BBW_TREND_PERCENTILE)
    is_mr_regime = bbw_pct_4h_mapped > BBW_MR_PERCENTILE
    is_neutral = ~is_trend_regime & ~is_mr_regime
    
    # Signal 1: HMA trend (15m)
    hma_signal = np.zeros(n)
    hma_signal[hma_15m > 0] = np.sign(close[hma_15m > 0] - hma_15m[hma_15m > 0])
    
    # Signal 2: Supertrend direction (15m)
    st_signal = st_dir_15m
    
    # Signal 3: RSI momentum (regime-dependent)
    rsi_signal = np.zeros(n)
    
    # Trend regime RSI
    trend_long_mask = is_trend_regime & (rsi_15m > RSI_TREND_LONG)
    trend_short_mask = is_trend_regime & (rsi_15m < RSI_TREND_SHORT)
    rsi_signal[trend_long_mask] = 1
    rsi_signal[trend_short_mask] = -1
    
    # Mean reversion regime RSI
    mr_long_mask = is_mr_regime & (rsi_15m < RSI_MR_LONG)
    mr_short_mask = is_mr_regime & (rsi_15m > RSI_MR_SHORT)
    rsi_signal[mr_long_mask] = 1
    rsi_signal[mr_short_mask] = -1
    
    # Neutral regime RSI
    neutral_long_mask = is_neutral & (rsi_15m > 55)
    neutral_short_mask = is_neutral & (rsi_15m < 45)
    rsi_signal[neutral_long_mask] = 1
    rsi_signal[neutral_short_mask] = -1
    
    # Signal 4: Bollinger position
    bb_signal = np.zeros(n)
    bb_signal[close < bb_lower_15m] = 1
    bb_signal[close > bb_upper_15m] = -1
    
    # === ENSEMBLE VOTING ===
    votes_long = np.zeros(n)
    votes_short = np.zeros(n)
    
    # HMA vote (weighted by 4h trend alignment)
    hma_aligned_long = (hma_signal == 1) & (trend_4h >= 0)
    hma_aligned_short = (hma_signal == -1) & (trend_4h <= 0)
    votes_long[hma_aligned_long] += 1.5
    votes_short[hma_aligned_short] += 1.5
    
    # Supertrend vote
    votes_long[st_signal == 1] += 1.0
    votes_short[st_signal == -1] += 1.0
    
    # RSI vote
    votes_long[rsi_signal == 1] += 1.0
    votes_short[rsi_signal == -1] += 1.0
    
    # Bollinger vote (mean reversion only)
    votes_long[is_mr_regime & (bb_signal == 1)] += 0.5
    votes_short[is_mr_regime & (bb_signal == -1)] += 0.5
    
    # Calculate confidence
    total_votes = np.maximum(votes_long, votes_short)
    confidence = np.clip(total_votes / 4.0, 0, 1)
    
    # Generate final signals with discrete levels
    long_mask = (votes_long >= 2.5) & (votes_long > votes_short)
    short_mask = (votes_short >= 2.5) & (votes_short > votes_long)
    
    # Apply regime-based sizing
    signals[long_mask & is_trend_regime] = SIZE_HIGH
    signals[long_mask & ~is_trend_regime] = SIZE_LOW
    signals[short_mask & is_trend_regime] = -SIZE_HIGH
    signals[short_mask & ~is_trend_regime] = -SIZE_LOW
    
    # Apply ATR filter (no trades in extremely high volatility)
    atr_pct = atr_15m / close
    high_vol_mask = atr_pct > np.percentile(atr_pct[~np.isnan(atr_pct)], 95)
    signals[high_vol_mask] = 0
    
    # Warmup period
    signals[:first_valid] = 0
    
    # Ensure no NaN values
    signals = np.nan_to_num(signals, nan=0.0)
    
    return signals