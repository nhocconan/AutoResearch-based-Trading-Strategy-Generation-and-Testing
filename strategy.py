#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
# - Daily Camarilla pivot levels (R1-R4, S1-S4) calculated from previous 1d OHLC
# - Long when price breaks above R3 with volume > 1.5x 20-day average volume
# - Short when price breaks below S3 with volume > 1.5x 20-day average volume
# - Exit when price returns to the pivot point (midpoint) or opposite S/R level
# - Position size: 0.25 (25%) to manage drawdown in volatile markets
# - Designed to capture institutional breakout moves with volume confirmation
# - Target: 20-50 trades/year to avoid excessive fee drag on 6h timeframe

name = "6h_Camarilla_R3_S3_Breakout_Volume_v1"
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
    
    # Get 1d data for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # R1 = Pivot * 2 - L, S1 = Pivot * 2 - H
    # R2 = Pivot + (H - L), S2 = Pivot - (H - L)
    # R3 = H + 2*(H - L), S3 = L - 2*(H - L)
    # R4 = H + 3*(H - L), S4 = L - 3*(H - L)
    pivot_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1_1d = pivot_1d * 2 - df_1d['low']
    s1_1d = pivot_1d * 2 - df_1d['high']
    r2_1d = pivot_1d + (df_1d['high'] - df_1d['low'])
    s2_1d = pivot_1d - (df_1d['high'] - df_1d['low'])
    r3_1d = df_1d['high'] + 2 * (df_1d['high'] - df_1d['low'])
    s3_1d = df_1d['low'] - 2 * (df_1d['high'] - df_1d['low'])
    r4_1d = df_1d['high'] + 3 * (df_1d['high'] - df_1d['low'])
    s4_1d = df_1d['low'] - 3 * (df_1d['high'] - df_1d['low'])
    
    # Pivot point (midpoint) for exit
    midpoint_1d = pivot_1d
    
    # Align Camarilla levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d.values)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d.values)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for volume MA (20) + alignment buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(midpoint_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-day average volume
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: price breaks above R3 with volume confirmation
            if close[i] > r3_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below S3 with volume confirmation
            elif close[i] < s3_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to midpoint or drops below S3 (reversal)
            if close[i] <= midpoint_1d_aligned[i] or close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to midpoint or rises above R3 (reversal)
            if close[i] >= midpoint_1d_aligned[i] or close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals