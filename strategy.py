#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Camarilla levels from 1d provide institutional support/resistance derived from prior day's range.
# Breakout at R3/S3 with volume spike confirms institutional participation on 12h chart.
# 1d EMA34 trend filter ensures alignment with higher timeframe direction.
# Works in both bull and bear markets by following 1d trend. Target: 50-150 trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (R3, S3) and EMA34 for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = H + 1.1*(L-H)/6, S3 = L - 1.1*(H-L)/6
    camarilla_range = high_1d - low_1d
    r3_1d = high_1d + 1.1 * camarilla_range / 6.0
    s3_1d = low_1d - 1.1 * camarilla_range / 6.0
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Camarilla and 1d EMA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with volume spike AND price > 1d EMA34 (bullish trend)
            if (close[i] > r3_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 with volume spike AND price < 1d EMA34 (bearish trend)
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla S3 OR below 1d EMA34 (trend change)
            if close[i] < s3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 OR above 1d EMA34 (trend change)
            if close[i] > r3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals