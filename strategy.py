#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 1w HTF for EMA50 trend alignment
- Donchian channels calculated from prior 4h OHLC (upper/lower 20-period)
- Breakout logic: long when price closes above upper band with volume spike and uptrend,
                  short when price closes below lower band with volume spike and downtrend
- Trend filter: only long when 4h close > 1w EMA50, only short when 4h close < 1w EMA50
- Volume confirmation: current 4h volume > 2.0 * 20-period 4h volume MA to capture institutional interest
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
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
    
    # Calculate 4h Donchian channels (20-period) from PRIOR completed bar
    # Shift by 1 to avoid look-ahead: use only data up to i-1 for bar i
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    # Calculate rolling max/min on shifted data
    upper_band = pd.Series(high_shifted).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_shifted).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 4h close vs 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need 20-period Donchian and 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND uptrend AND volume spike
            if close[i] > upper_band[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band AND downtrend AND volume spike
            elif close[i] < lower_band[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to lower band (mean reversion) or reverse signal
            if close[i] <= lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to upper band (mean reversion) or reverse signal
            if close[i] >= upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_1wEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0