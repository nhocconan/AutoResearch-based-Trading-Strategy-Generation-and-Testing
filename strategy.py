#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise intraday support/resistance. 12h EMA50 ensures alignment with higher timeframe trend.
# Volume spike (>1.8x 20 EMA) confirms institutional participation. Discrete sizing 0.25 limits risk and reduces fee drag.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 75-200 trades over 4 years.

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    close_12h = pd.Series(df_12h['close'])
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (completed 12h bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8 x 20-period EMA
        volume_confirm = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot point OR trend changes OR volume drops
            camarilla_pivot = (high_1d[i//16] + low_1d[i//16] + close_1d[i//16]) / 3.0 if i//16 < len(high_1d) else camarilla_pivot
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, camarilla_pivot))
            midpoint = camarilla_pivot_aligned[i] if not np.isnan(camarilla_pivot_aligned[i]) else (r3_aligned[i] + s3_aligned[i]) / 2.0
            
            if (close[i] < midpoint or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot point OR trend changes OR volume drops
            camarilla_pivot = (high_1d[i//16] + low_1d[i//16] + close_1d[i//16]) / 3.0 if i//16 < len(high_1d) else camarilla_pivot
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, camarilla_pivot))
            midpoint = camarilla_pivot_aligned[i] if not np.isnan(camarilla_pivot_aligned[i]) else (r3_aligned[i] + s3_aligned[i]) / 2.0
            
            if (close[i] > midpoint or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals