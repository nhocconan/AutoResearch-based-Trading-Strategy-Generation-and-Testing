#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses 1d timeframe (primary) and 1w HTF for EMA50 trend alignment (proven pattern from DB)
- Donchian levels calculated from previous completed 1d bar's high/low (based on prior 1d candle)
- Long when price breaks above Donchian upper AND price > 1w EMA50 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below Donchian lower AND price < 1w EMA50 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the midpoint of the Donchian channel (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 30-100 total trades over 4 years (7-25/year) as per 1d timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Use previous completed 1d bar's OHLC for Donchian calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Shift by 1 to use previous completed 1d bar's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar has no previous bar, set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate 1w EMA50 for trend filter (using previous completed 1w bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian levels from previous completed 1d bar's high/low
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    # We calculate on previous bar data to avoid look-ahead
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Use data up to previous bar (i-1) for Donchian calculation
        start_idx = i - lookback
        end_idx = i  # exclusive, so we use [start_idx:end_idx] which is [i-lookback, i-1]
        if start_idx >= 0 and not (np.isnan(prev_high[start_idx:end_idx]).any() or np.isnan(prev_low[start_idx:end_idx]).any()):
            upper[i] = np.max(prev_high[start_idx:end_idx])
            lower[i] = np.min(prev_low[start_idx:end_idx])
    
    # Midpoint for exit
    midpoint = (upper + lower) / 2.0
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, lookback)  # Need 1w EMA50, volume MA(20), and Donchian lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND uptrend AND volume confirmation
            if close[i] > upper[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND downtrend AND volume confirmation
            elif close[i] < lower[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Donchian midpoint
            if close[i] < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Donchian midpoint
            if close[i] > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0