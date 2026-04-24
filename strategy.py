#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike.
- Uses 4h timeframe (primary) and 12h HTF for trend alignment (proven pattern from DB)
- Camarilla pivot levels from previous 1d OHLC (structure-based support/resistance)
- Long when price breaks above Camarilla R3 AND price > 12h EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below Camarilla S3 AND price < 12h EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to Camarilla pivot point (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) as per 4h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d OHLC for Camarilla pivots (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least 1 day of data
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous 1d bar)
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # S3 = close - (high-low)*1.1/4
    # Pivot = (high + low + close)/3
    camarilla_high = (high_1d + low_1d + close_1d) / 3.0 + (high_1d - low_1d) * 1.1 / 4.0  # R3
    camarilla_low = (high_1d + low_1d + close_1d) / 3.0 - (high_1d - low_1d) * 1.1 / 4.0   # S3
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0  # PP
    
    # Align Camarilla levels to 4h timeframe (previous 1d Camarilla available at open)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 12h EMA34
    uptrend = close > ema_34_12h_aligned
    downtrend = close < ema_34_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 12h EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND uptrend AND volume confirmation
            if close[i] > camarilla_high_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_low_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Camarilla pivot point
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Camarilla pivot point
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_12hEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0