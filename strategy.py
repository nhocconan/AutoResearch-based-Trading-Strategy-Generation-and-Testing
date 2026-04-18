#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high/low from daily data (lookback 5 days)
    weekly_high = np.full_like(close_1d, np.nan)
    weekly_low = np.full_like(close_1d, np.nan)
    
    for i in range(4, len(close_1d)):
        weekly_high[i] = np.max(high_1d[i-4:i+1])
        weekly_low[i] = np.min(low_1d[i-4:i+1])
    
    # Calculate weekly pivot points
    # Pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    # R2 = P + (H - L), S2 = P - (H - L)
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly data to 6h timeframe
    weekly_high_6h = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_6h = align_htf_to_ltf(prices, df_1d, weekly_low)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 6h volume spike (volume > 2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_6h[i]) or np.isnan(weekly_low_6h[i]) or 
            np.isnan(weekly_pivot_6h[i]) or np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above S3 with volume spike (mean reversion from extreme oversold)
            if close[i] > s3_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R3 with volume spike (mean reversion from extreme overbought)
            elif close[i] < r3_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 (take profit) or breaks below weekly low (stop)
            if close[i] >= r1_6h[i] or close[i] < weekly_low_6h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 (take profit) or breaks above weekly high (stop)
            if close[i] <= s1_6h[i] or close[i] > weekly_high_6h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_S3_R1_MeanReversion_VolumeSpike"
timeframe = "6h"
leverage = 1.0