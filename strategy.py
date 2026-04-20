#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Weekly Donchian breakout with weekly pivot direction and volume confirmation
# Weekly Donchian(20) provides major support/resistance levels
# Weekly pivot determines bias (above/below weekly pivot)
# Volume spike confirms institutional participation
# Target: 12-30 trades per year to minimize fee decay while capturing major moves
# Works in bull/bear by following weekly structure

name = "6h_WeeklyDonchian_Pivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # === Weekly Donchian(20) channels ===
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donch_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_weekly, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_weekly, donch_low)
    
    # === Weekly pivot points (based on prior week) ===
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # === 6h volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 * 6h = 6 days
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = prices['high'].iloc[i]
        low_val = prices['low'].iloc[i]
        dh = donch_high_aligned[i]
        dl = donch_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(dh) or np.isnan(dl) or np.isnan(vol_ratio_val) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian high + above weekly pivot + volume spike
            if (close_val > dh and 
                close_val > pivot_aligned[i] and 
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low + below weekly pivot + volume spike
            elif (close_val < dl and 
                  close_val < pivot_aligned[i] and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below weekly Donchian low or below weekly S1
            if close_val < dl or close_val < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above weekly Donchian high or above weekly R1
            if close_val > dh or close_val > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals