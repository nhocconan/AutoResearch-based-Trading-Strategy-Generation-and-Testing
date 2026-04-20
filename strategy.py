# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly ATR for volatility filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # === Daily: High, Low, Close for Camarilla calculation ===
    high_d = prices['high'].values
    low_d = prices['low'].values
    close_d = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate previous day's Camarilla levels
    prev_high = np.concatenate([[np.nan], high_d[:-1]])
    prev_low = np.concatenate([[np.nan], low_d[:-1]])
    prev_close = np.concatenate([[np.nan], close_d[:-1]])
    
    # Camarilla R1 and S1 (using previous day's range)
    prev_range = prev_high - prev_low
    r1 = prev_close + (prev_range * 1.1 / 12)
    s1 = prev_close - (prev_range * 1.1 / 12)
    
    # Volume ratio (current vs 20-period average) with min_periods
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr14_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and low weekly volatility
            if (close_d[i] > r1[i] and 
                vol_ratio[i] > 1.8 and 
                atr14_1w_aligned[i] < np.nanpercentile(atr14_1w_aligned[:i+1], 70)):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and low weekly volatility
            elif (close_d[i] < s1[i] and 
                  vol_ratio[i] > 1.8 and 
                  atr14_1w_aligned[i] < np.nanpercentile(atr14_1w_aligned[:i+1], 70)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls back below R1 or volatility increases
            if (close_d[i] < r1[i] or 
                atr14_1w_aligned[i] > np.nanpercentile(atr14_1w_aligned[:i+1], 85)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1 or volatility increases
            if (close_d[i] > s1[i] or 
                atr14_1w_aligned[i] > np.nanpercentile(atr14_1w_aligned[:i+1], 85)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals