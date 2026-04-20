#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R2S2_Breakout_Volume_12h_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # === 1d Previous Day Prices for Pivot Calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first values to current day's values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic pivot (base for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels: R2, S2, R3, S3
    r2 = pivot + (range_val * 1.1 / 6)
    s2 = pivot - (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align to 12h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r2_val) or np.isnan(s2_val) or
            np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion at R2/S2 (strong support/resistance)
            if close_val < r2_val and close_val > r2_val * 0.995 and vol_ratio_val > 1.8:
                # Near R2, short with volume
                signals[i] = -0.25
                position = -1
            elif close_val > s2_val and close_val < s2_val * 1.005 and vol_ratio_val > 1.8:
                # Near S2, long with volume
                signals[i] = 0.25
                position = 1
            # Breakout at R3/S3 with strong volume
            elif close_val > r3_val and vol_ratio_val > 2.5:
                # Strong break above R3
                signals[i] = 0.25
                position = 1
            elif close_val < s3_val and vol_ratio_val > 2.5:
                # Strong break below S3
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to pivot or stop at S2
            if close_val < pivot_val or close_val < s2_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to pivot or stop at R2
            if close_val > pivot_val or close_val > r2_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals