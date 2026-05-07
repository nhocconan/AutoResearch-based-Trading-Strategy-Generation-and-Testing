#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (R1, S1, R3, S3)
    R1 = close_1d + range_1d * 1.1 / 12
    S1 = close_1d - range_1d * 1.1 / 12
    R3 = close_1d + range_1d * 1.1 / 4
    S3 = close_1d - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation (4h volume > 2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R1 with volume in 12h uptrend
            if close[i] > R1_aligned[i] and vol_condition and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume in 12h downtrend
            elif close[i] < S1_aligned[i] and vol_condition and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to S1 or trend breaks
            if close[i] < S1_aligned[i] or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to R1 or trend breaks
            if close[i] > R1_aligned[i] or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# - Uses 12h EMA50 for trend alignment (avoids counter-trend trades)
# - Enters long on break above Camarilla R1, short on break below S1
# - Requires volume > 2x 20-period average to confirm breakout strength
# - Exits when price returns to opposite Camarilla level (S1 for longs, R1 for shorts) or trend breaks
# - Position size 0.25 balances return potential with risk management
# - Designed for 20-50 trades/year to avoid excessive fee drag
# - Combines proven elements: Camarilla pivots (effective S/R), trend filtering, volume confirmation
# - Works in both bull and bear markets via trend filter
# - Avoids overtrading through strict entry conditions (breakout + volume + trend alignment)