# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot level S1/R1 bounce with volume confirmation and 1d EMA200 trend filter
# Works in bull market: price respects pivot support/resistance with volume confirmation
# Works in bear market: EMA200 filter prevents counter-trend trades, pivots act as magnet levels for mean reversion
# Low trade frequency expected due to strict pivot level requirements + volume + trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivot calculation and EMA200
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # R1 = C + (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    
    # Align daily pivot levels to 12h timeframe (no extra delay needed for pivot levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Calculate EMA200 on daily data for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # need volume MA and EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(r1_12h[i]) or 
            np.isnan(ema200_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price near S1 support (> S1 * 0.998 and < S1 * 1.002) with volume and above EMA200
            s1_lower = s1_12h[i] * 0.998
            s1_upper = s1_12h[i] * 1.002
            near_s1 = (close[i] >= s1_lower) and (close[i] <= s1_upper)
            
            if near_s1 and vol_confirmed and (close[i] > ema200_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price near R1 resistance (> R1 * 0.998 and < R1 * 1.002) with volume and below EMA200
            elif (close[i] >= r1_12h[i] * 0.998) and (close[i] <= r1_12h[i] * 1.002) and vol_confirmed and (close[i] < ema200_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price moves above pivot or below S1
            if (close[i] > pivot_12h[i]) or (close[i] < s1_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price moves below pivot or above R1
            if (close[i] < pivot_12h[i]) or (close[i] > r1_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_S1R1_Volume_EMA200"
timeframe = "12h"
leverage = 1.0