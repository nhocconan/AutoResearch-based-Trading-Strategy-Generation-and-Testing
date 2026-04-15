#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian high + weekly pivot resistance broken + volume spike
# Short when price breaks below Donchian low + weekly pivot support broken + volume spike
# Weekly pivot uses prior week's OHLC to calculate classic pivot points (P, R1/R2/R3, S1/S2/S3)
# Designed to capture breakouts with institutional interest (volume) and structural bias (weekly pivot)
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly OHLC for pivot points
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    # R3 = H + 2*(P-L), S3 = L - 2*(H-P)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    r3 = high_w + 2 * (pivot - low_w)
    s3 = low_w - 2 * (high_w - pivot)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: current > 2.0x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: break above Donchian high AND above weekly R3 + volume spike
        if (close[i] > donch_high[i] and 
            close[i] > r3_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: break below Donchian low AND below weekly S3 + volume spike
        elif (close[i] < donch_low[i] and 
              close[i] < s3_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns inside Donchian channel or volume drops
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < donch_high[i] or volume[i] <= vol_threshold[i])) or
               (signals[i-1] == -0.25 and (close[i] > donch_low[i] or volume[i] <= vol_threshold[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0