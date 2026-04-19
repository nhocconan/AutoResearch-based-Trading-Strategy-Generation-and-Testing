#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_R1S1_Breakout_Volume_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pivot - df_1d['low']
    s1 = 2 * pivot - df_1d['high']
    r2 = pivot + (df_1d['high'] - df_1d['low'])
    s2 = pivot - (df_1d['high'] - df_1d['low'])
    r3 = df_1d['high'] + 2 * (pivot - df_1d['low'])
    s3 = df_1d['low'] - 2 * (df_1d['high'] - pivot)
    
    # Align pivot levels to 6h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: current 6h volume > 1.5x average 6h volume
    # Calculate average 6h volume over last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and uptrend
            if (close[i] > r1_aligned[i] and 
                volume_filter and 
                close[i] > ema_34_aligned[i]):  # Uptrend filter
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume_filter and 
                  close[i] < ema_34_aligned[i]):  # Downtrend filter
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls back below pivot or trend changes
            if (close[i] < pivot_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises back above pivot or trend changes
            if (close[i] > pivot_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals