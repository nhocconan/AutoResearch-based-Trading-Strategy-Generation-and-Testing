#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d, HTF: 1w for EMA50 trend alignment.
- Donchian breakout from prior 20 periods: long when close > upper band, short when close < lower band.
- Trend filter: only long when 1d close > 1w EMA50, only short when 1d close < 1w EMA50.
- Volume confirmation: current 1d volume > 1.5 * 20-period 1d volume MA (moderate filter).
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakout captures momentum.
- Exit: price reverts to midpoint of Donchian channel from prior 20 periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) from prior 20 periods (use completed 1d bar only)
    # Upper band = max(high, 20)
    # Lower band = min(low, 20)
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    middle_band = (upper_band + lower_band) / 2.0
    
    # Align Donchian levels to 1d timeframe (completed 1d bar only)
    upper_band_aligned = align_htf_to_ltf(prices, prices, upper_band)  # Same timeframe, no alignment needed but use helper for consistency
    lower_band_aligned = align_htf_to_ltf(prices, prices, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, prices, middle_band)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Trend filter: 1d close vs 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need 1w EMA50, Donchian(20), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(middle_band_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND uptrend AND volume spike
            if close[i] > upper_band_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band AND downtrend AND volume spike
            elif close[i] < lower_band_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle band or reverse signal
            if close[i] <= middle_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle band or reverse signal
            if close[i] >= middle_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0