#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeS
# Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA trend filter and volume spike.
# Long when price breaks above R1 with price > weekly EMA and volume > 2x volume MA.
# Short when price breaks below S1 with price < weekly EMA and volume > 2x volume MA.
# Exit when price reverts to the daily pivot point.
# Uses 1d timeframe with 1h trend filter for reduced whipsaw.
# Targets 15-25 trades/year to minimize fee drag and improve generalization.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeS"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Pivot point and support/resistance levels
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    
    # Camarilla levels
    R1 = close + (range_hl * 1.1 / 12)
    S1 = close - (range_hl * 1.1 / 12)
    R2 = close + (range_hl * 1.1 / 6)
    S2 = close - (range_hl * 1.1 / 6)
    R3 = close + (range_hl * 1.1 / 4)
    S3 = close - (range_hl * 1.1 / 4)
    
    # Weekly EMA for trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with price > weekly EMA and volume > 2x MA
            if close[i] > R1[i] and close[i] > weekly_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with price < weekly EMA and volume > 2x MA
            elif close[i] < S1[i] and close[i] < weekly_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back below pivot point
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back above pivot point
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals