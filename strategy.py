#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_Pivot_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for trend filter (EMA34) and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly Camarilla pivot levels (R1, S1, R2, S2, R3, S3)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    
    R1 = pivot + (range_w * 1.1 / 12)
    S1 = pivot - (range_w * 1.1 / 12)
    R2 = pivot + (range_w * 1.1 / 6)
    S2 = pivot - (range_w * 1.1 / 6)
    R3 = pivot + (range_w * 1.1 / 4)
    S3 = pivot - (range_w * 1.1 / 4)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume average (20-period) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_aligned[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current volume > 1.5x daily average
        vol_confirm = volume[i] > (vol_avg_aligned[i] * 1.5)
        
        if position == 0:
            # Long when price breaks above R3 with volume and above daily EMA34
            if (close[i] > R3_aligned[i] and 
                vol_confirm and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 with volume and below daily EMA34
            elif (close[i] < S3_aligned[i] and 
                  vol_confirm and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below R1 or volume drops
            if (close[i] < R1_aligned[i] or 
                volume[i] < (vol_avg_aligned[i] * 0.5)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above S1 or volume drops
            if (close[i] > S1_aligned[i] or 
                volume[i] < (vol_avg_aligned[i] * 0.5)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals