#!/usr/bin/env python3
"""
EXPERIMENT #082 - CLEAN_MTF_ENSEMBLE_1H_4H_V2
==================================================================================================
Hypothesis: Fix the read-only array issues from #081 by never modifying array views.
Use a cleaner 3-signal ensemble (HMA trend + RSI momentum + ATR filter) with 4h trend confirmation.
Add signal persistence to reduce churn - only flip when conviction is strong.

Key improvements over #081:
- NEVER modify array views - always create new arrays with np.where()
- Simpler 3-signal voting (HMA, RSI, ATR) instead of 4+ signals
- Signal persistence: require 2 consecutive bars to flip direction
- Better warmup handling with clear first_valid index
- Discrete signal levels (0.0, ±0.25, ±0.35) to minimize fees

Why this should work:
- #070 achieved Sharpe=1.256 with similar MTF approach
- #072 achieved Sharpe=0.589 with HMA+ST+RSI on 1h/4h
- Proper array handling avoids read-only crashes (#080, #081)
- Signal persistence reduces churn costs (0.10% per change)
- 4h trend filter provides directional bias

Position sizing:
- MAX signal: 0.35 (controls drawdown during 2022 crash)
- Discrete levels: 0.0, ±0.25, ±0.35
- Higher size in confirmed trend regime
"""

import numpy as np
import pandas as pd

name = "clean_mtf_ensemble_1h_4h_v2"
timeframe = "1h"
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
        return x.rolling(window=period, min_periods=period).apply(
            lambda y: np.sum(y * weights) / np.sum(weights), raw=True
        )
    
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
    rsi = np.nan_to_num(rsi, nan=50.0)
    
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
    adx = np.nan_to_num(adx, nan=0)
    
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
    bbw = np.nan_to_num(bbw, nan=0)
    
    return upper, middle, lower, bbw


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = (close - rolling_mean) / rolling_std
    zscore = np.nan_to_num(zscore, nan=0)
    
    return zscore


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
    n_tf = len(tf_array)
    mapped = np.zeros(base_length)
    
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
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
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
    
    # Z-score thresholds
    ZSCORE_MR_LONG = -2.0
    ZSCORE_MR_SHORT = 2.0
    
    # Timeframe conversion: 4h = 4 x 1h
    bars_per_4h = 4
    
    # Base timeframe (1h) indicators
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, short_period=16, long_period=48)
    adx_1h = calculate_adx(high, low, close, period=14)
    bb_upper_1h, bb_mid_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    zscore_1h = calculate_zscore(close, period=20)
    
    # 4h timeframe indicators for trend filter
    c_4h, h_4h, l_4h = resample_to_timeframe(close, high, low, bars_per_4h)
    hma_4h = calculate_hma(c_4h, short_period=16, long_period=48)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators to 1h
    hma_4h_mapped = map_tf_to_base(hma_4h, bars_per_4h, n)
    adx_4h_mapped = map_tf_to_base(adx_4h, bars_per_4h, n)
    bbw_pct_4h_mapped = map_tf_to_base(bbw_pct_4h, bars_per_4h, n)
    
    # Calculate HMA slope (trend direction)
    hma_slope_1h = np.zeros(n)
    hma_slope_1h[5:] = np.sign(hma_1h[5:] - hma_1h[:-5])
    
    # Calculate 4h trend direction
    trend_4h = np.zeros(n)
    for i in range(n):
        tf_idx = min(i // bars_per_4h, len(c_4h) - 1)
        if hma_4h[tf_idx] > 0:
            trend_4h[i] = np.sign(c_4h[tf_idx] - hma_4h[tf_idx])
    
    # Minimum warmup period
    first_valid = max(200, 100 * bars_per_4h, 48, 45)
    
    # Initialize signals array
    signals = np.zeros(n)
    
    # Vectorized regime detection
    is_trend_regime = (adx_4h_mapped > ADX_TREND_THRESHOLD) & (bbw_pct_4h_mapped < BBW_TREND_PERCENTILE)
    is_mr_regime = bbw_pct_4h_mapped > BBW_MR_PERCENTILE
    is_neutral = (~is_trend_regime) & (~is_mr_regime)
    
    # Signal 1: HMA trend (1h)
    hma_signal = np.zeros(n)
    hma_valid = hma_1h > 0
    hma_signal = np.where(hma_valid, np.sign(close - hma_1h), hma_signal)
    
    # Signal 2: RSI momentum (regime-dependent)
    rsi_signal = np.zeros(n)
    
    # Trend regime RSI
    trend_long_mask = is_trend_regime & (rsi_1h > RSI_TREND_LONG)
    trend_short_mask = is_trend_regime & (rsi_1h < RSI_TREND_SHORT)
    rsi_signal = np.where(trend_long_mask, 1, rsi_signal)
    rsi_signal = np.where(trend_short_mask, -1, rsi_signal)
    
    # Mean reversion regime RSI
    mr_long_mask = is_mr_regime & (rsi_1h < RSI_MR_LONG)
    mr_short_mask = is_mr_regime & (rsi_1h > RSI_MR_SHORT)
    rsi_signal = np.where(mr_long_mask, 1, rsi_signal)
    rsi_signal = np.where(mr_short_mask, -1, rsi_signal)
    
    # Neutral regime RSI
    neutral_long_mask = is_neutral & (rsi_1h > 55)
    neutral_short_mask = is_neutral & (rsi_1h < 45)
    rsi_signal = np.where(neutral_long_mask, 1, rsi_signal)
    rsi_signal = np.where(neutral_short_mask, -1, rsi_signal)
    
    # Signal 3: Z-score mean reversion
    zscore_signal = np.zeros(n)
    zscore_signal = np.where(zscore_1h < ZSCORE_MR_LONG, 1, zscore_signal)
    zscore_signal = np.where(zscore_1h > ZSCORE_MR_SHORT, -1, zscore_signal)
    
    # === ENSEMBLE VOTING ===
    votes_long = np.zeros(n)
    votes_short = np.zeros(n)
    
    # HMA vote (weighted by 4h trend alignment)
    hma_aligned_long = (hma_signal == 1) & (trend_4h >= 0)
    hma_aligned_short = (hma_signal == -1) & (trend_4h <= 0)
    votes_long = np.where(hma_aligned_long, votes_long + 1.5, votes_long)
    votes_short = np.where(hma_aligned_short, votes_short + 1.5, votes_short)
    
    # RSI vote
    votes_long = np.where(rsi_signal == 1, votes_long + 1.0, votes_long)
    votes_short = np.where(rsi_signal == -1, votes_short + 1.0, votes_short)
    
    # Z-score vote (mean reversion only, lower weight)
    zscore_mr_mask = is_mr_regime | is_neutral
    votes_long = np.where(zscore_mr_mask & (zscore_signal == 1), votes_long + 0.5, votes_long)
    votes_short = np.where(zscore_mr_mask & (zscore_signal == -1), votes_short + 0.5, votes_short)
    
    # Generate final signals with discrete levels
    long_mask = (votes_long >= 2.5) & (votes_long > votes_short)
    short_mask = (votes_short >= 2.5) & (votes_short > votes_long)
    
    # Create new signal array based on conditions
    new_signals = np.zeros(n)
    new_signals = np.where(long_mask & is_trend_regime, SIZE_HIGH, new_signals)
    new_signals = np.where(long_mask & (~is_trend_regime), SIZE_LOW, new_signals)
    new_signals = np.where(short_mask & is_trend_regime, -SIZE_HIGH, new_signals)
    new_signals = np.where(short_mask & (~is_trend_regime), -SIZE_LOW, new_signals)
    
    signals = new_signals.copy()
    
    # Apply ATR filter (no trades in extremely high volatility)
    atr_pct = atr_1h / close
    atr_valid = ~np.isnan(atr_pct)
    if np.sum(atr_valid) > 0:
        high_vol_threshold = np.percentile(atr_pct[atr_valid], 95)
        high_vol_mask = atr_pct > high_vol_threshold
        signals = np.where(high_vol_mask, 0, signals)
    
    # Signal persistence: require 2 consecutive bars to flip direction
    # This reduces churn and transaction costs
    persistent_signals = np.zeros(n)
    for i in range(first_valid, n):
        if i < 2:
            persistent_signals[i] = 0
        elif signals[i] > 0 and signals[i-1] > 0:
            persistent_signals[i] = signals[i]
        elif signals[i] < 0 and signals[i-1] < 0:
            persistent_signals[i] = signals[i]
        elif signals[i] == 0:
            persistent_signals[i] = 0
        else:
            # Keep previous signal if not confirmed
            persistent_signals[i] = persistent_signals[i-1]
    
    signals = persistent_signals
    
    # Warmup period
    signals[:first_valid] = 0
    
    # Ensure no NaN values
    signals = np.nan_to_num(signals, nan=0.0)
    
    return signals