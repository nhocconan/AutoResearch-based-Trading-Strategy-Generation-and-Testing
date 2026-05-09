#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_1wTrendFilter_Volume
# Strategy: Trade Camarilla R1/S1 breakouts on 1d with 1w trend filter and volume confirmation
# Long when price breaks above R1 and close > 1w EMA50 and volume > 1.5x average
# Short when price breaks below S1 and close < 1w EMA50 and volume > 1.5x average
# Exit when price reverts to Camarilla pivot or opposite level
# Uses 1w trend to avoid counter-trend trades and volume to confirm breakout strength
# Designed for 1d timeframe with selective entries to minimize trade frequency

name = "1d_Camarilla_R1S1_Breakout_1wTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i]) or vol_avg[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        if i == 0:
            continue
            
        ph = high[i-1]  # Previous day high
        pl = low[i-1]   # Previous day low
        pc = close[i-1] # Previous day close
        
        # Camarilla levels
        R1 = pc + (ph - pl) * 1.1 / 12
        S1 = pc - (ph - pl) * 1.1 / 12
        pivot = (ph + pl + pc) / 3
        
        if position == 0:
            # Enter long: price breaks above R1 with trend and volume confirmation
            if (close[i] > R1 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with trend and volume confirmation
            elif (close[i] < S1 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot or below S1
            if close[i] <= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot or above R1
            if close[i] >= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals