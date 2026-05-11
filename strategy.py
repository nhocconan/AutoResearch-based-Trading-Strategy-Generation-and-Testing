#!/usr/bin/env python3
name = "6h_1d_Volume_Weighted_Price_Action"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily VWAP calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Daily trend: price above/below VWAP
    trend_up_1d = df_1d['close'].values > vwap_values
    trend_down_1d = df_1d['close'].values < vwap_values
    
    # Align to 6h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # Volume filter: current volume > 2.0x 20-period average (tighter filter)
    vol_ma20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i < 19:
            vol_ma20[i] = vol_sum / (i + 1) if i > 0 else 0
        else:
            vol_ma20[i] = vol_sum / 20
    
    # Price position relative to VWAP bands (1% bands)
    vwap_upper = vwap_aligned * 1.01
    vwap_lower = vwap_aligned * 0.99
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(vwap_upper[i]) or np.isnan(vwap_lower[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP upper band in daily uptrend with volume surge
            if (close[i] > vwap_upper[i] and 
                trend_up_aligned[i] and 
                volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP lower band in daily downtrend with volume surge
            elif (close[i] < vwap_lower[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP lower band or trend changes
            if (close[i] < vwap_lower[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP upper band or trend changes
            if (close[i] > vwap_upper[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals