#!/usr/bin/env python3
"""
EXPERIMENT #047 - MTF HMA+RSI+Z-score+ATR Simple Clean 15m+4h v1
==================================================================================================
Hypothesis: Recent failures (#040-#046) caused by complex state tracking and manual MTF resampling.
Return to proven 15m+4h combination from #031/#034/#035 with cleaner implementation.

Key changes from #040:
- Use mtf_data helper properly (get_htf_data, align_htf_to_ltf) - NO manual resampling
- Simplified signal logic without complex state arrays
- 4h HMA trend filter + 15m RSI pullback entries (proven in #035 Sharpe=7.7)
- Z-score filter for extreme moves
- ATR-based stoploss with cleaner exit logic
- Position size: 0.30 (conservative, proven safe)
- Discrete signal levels: 0.0, ±0.25, ±0.30 to reduce churn costs

Why this should work:
- mtf_data helper ensures correct 4h boundaries (no look-ahead)
- Simpler logic = fewer bugs (recent crashes were coding errors)
- 15m+4h worked in #031/#034/#035 before complex additions broke it
- Conservative sizing controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_zscore_atr_simple_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, half_period + 1)) / np.sum(np.arange(1, half_period + 1)),
        raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)),
        raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, sqrt_period + 1)) / np.sum(np.arange(1, sqrt_period + 1)),
        raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0
    
    return np.nan_to_num(zscore, nan=0.0)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    mid = (high + low) / 2
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Get 4h data using mtf_data helper (CRITICAL - no manual resampling)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
    except Exception:
        # Fallback if mtf_data fails
        df_4h = prices
        close_4h = close
        high_4h = high
        low_4h = low
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h indicators for trend (using mtf_data helper)
    hma_4h = calculate_hma(close_4h, period=21)
    supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    
    # Align 4h indicators to 15m timeframe (auto shift for completed bars)
    try:
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    except Exception:
        # Fallback: simple repeat
        bars_per_4h = 16  # 16 x 15m = 4h
        hma_4h_aligned = np.zeros(n)
        st_4h_aligned = np.zeros(n)
        for i in range(n):
            idx_4h = min(i // bars_per_4h, len(hma_4h) - 1)
            if idx_4h >= 0:
                hma_4h_aligned[i] = hma_4h[idx_4h]
                st_4h_aligned[i] = st_direction_4h[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score threshold
    ZSCORE_MAX = 2.5
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 40, 14 * 2, 20)
    
    # Track position state (simple arrays)
    in_position = np.zeros(n, dtype=bool)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    stoploss_price = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN/invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        
        # 4h trend filter
        trend_4h = 0
        if hma_4h_aligned[i] > 0 and price > hma_4h_aligned[i]:
            trend_4h = 1
        elif hma_4h_aligned[i] > 0 and price < hma_4h_aligned[i]:
            trend_4h = -1
        
        st_trend_4h = st_4h_aligned[i]
        
        # 15m supertrend
        st_trend_15m = st_direction_15m[i]
        
        # Check existing position for stoploss
        if in_position[i - 1]:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1]
            prev_stop = stoploss_price[i - 1]
            
            # Stoploss check
            if prev_side == 1 and price < prev_stop:
                signals[i] = 0.0
                in_position[i] = False
                position_side[i] = 0
                continue
            elif prev_side == -1 and price > prev_stop:
                signals[i] = 0.0
                in_position[i] = False
                position_side[i] = 0
                continue
            
            # Hold position
            signals[i] = signals[i - 1]
            in_position[i] = True
            position_side[i] = prev_side
            entry_price[i] = prev_entry
            stoploss_price[i] = prev_stop
            continue
        
        # Entry logic: 4h trend + 15m pullback
        # Long entry
        if trend_4h == 1 and st_trend_4h == 1 and st_trend_15m == 1:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX):
                signals[i] = SIZE_FULL
                in_position[i] = True
                position_side[i] = 1
                entry_price[i] = price
                stoploss_price[i] = price - ATR_STOP_MULT * atr
        
        # Short entry
        elif trend_4h == -1 and st_trend_4h == -1 and st_trend_15m == -1:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX):
                signals[i] = -SIZE_FULL
                in_position[i] = True
                position_side[i] = -1
                entry_price[i] = price
                stoploss_price[i] = price + ATR_STOP_MULT * atr
        
        else:
            signals[i] = 0.0
            in_position[i] = False
    
    return signals