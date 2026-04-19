#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_Trend"
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels on 1d data
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # Pivot = (H+L+C)/3
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    pivot = (H + L + C) / 3.0
    range_hl = H - L
    R1 = C + range_hl * 1.1 / 12.0
    S1 = C - range_hl * 1.1 / 12.0
    R2 = C + range_hl * 1.1 / 6.0
    S2 = C - range_hl * 1.1 / 6.0
    R3 = C + range_hl * 1.1 / 4.0
    S3 = C - range_hl * 1.1 / 4.0
    R4 = C + range_hl * 1.1 / 2.0
    S4 = C - range_hl * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 1d close > 50 EMA for long, < 50 EMA for short
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(vol_ma_12h[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 20-period average
        volume_filter = vol_ma_12h[i] > 0 and volume[i] > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume and uptrend
            if (close[i] > R1_aligned[i] and 
                volume_filter and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume and downtrend
            elif (close[i] < S1_aligned[i] and 
                  volume_filter and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below S1 or trend changes
            if (close[i] < S1_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above R1 or trend changes
            if (close[i] > R1_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals