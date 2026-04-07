#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Camarilla formulas
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.0 * (prev_high - prev_low)
    H2 = prev_close + 0.5 * (prev_high - prev_low)
    H1 = prev_close + 0.25 * (prev_high - prev_low)
    L1 = prev_close - 0.25 * (prev_high - prev_low)
    L2 = prev_close - 0.5 * (prev_high - prev_low)
    L3 = prev_close - 1.0 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align all levels to 12h timeframe (shifted by 1 day for lookback)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    H2_12h = align_htf_to_ltf(prices, df_1d, H2)
    H1_12h = align_htf_to_ltf(prices, df_1d, H1)
    L1_12h = align_htf_to_ltf(prices, df_1d, L1)
    L2_12h = align_htf_to_ltf(prices, df_1d, L2)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any pivot level is not ready
        if (np.isnan(H4_12h[i]) or np.isnan(H3_12h[i]) or np.isnan(H2_12h[i]) or 
            np.isnan(H1_12h[i]) or np.isnan(L1_12h[i]) or np.isnan(L2_12h[i]) or 
            np.isnan(L3_12h[i]) or np.isnan(L4_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L2 (strong support broken)
            if close[i] < L2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H2 (strong resistance broken)
            if close[i] > H2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume must be present for any entry
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above H3 with volume (bullish breakout)
            if close[i] > H3_12h[i] and close[i-1] <= H3_12h[i-1]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 with volume (bearish breakout)
            elif close[i] < L3_12h[i] and close[i-1] >= L3_12h[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals