# 6h_12h_Camarilla_R3S3_Fade_R4S4_Breakout
# Fade at R3/S3 with volume confirmation, breakout at R4/S4
# Works in both bull/bear: mean reversion in range, breakout in trend
# Uses 12h Camarilla for 6h timeframe
# Target: 50-150 trades over 4 years (12-37/year)
# Size: 0.25

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Camarilla_R3S3_Fade_R4S4_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h Camarilla Pivot Points (previous day) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    # Classic pivot (same for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 4)  # Strong resistance
    s3 = pivot - (range_val * 1.1 / 4)  # Strong support
    r4 = pivot + (range_val * 1.1 / 2)  # Breakout level
    s4 = pivot - (range_val * 1.1 / 2)  # Breakdown level
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r3_val) or 
            np.isnan(s3_val) or np.isnan(r4_val) or 
            np.isnan(s4_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Fade at R3/S3: mean reversion
            if close_val < r3_val and close_val > r3_val * 0.998 and vol_ratio_val > 1.5:
                # Near R3, short with volume
                signals[i] = -0.25
                position = -1
            elif close_val > s3_val and close_val < s3_val * 1.002 and vol_ratio_val > 1.5:
                # Near S3, long with volume
                signals[i] = 0.25
                position = 1
            # Breakout at R4/S4: trend continuation
            elif close_val > r4_val and vol_ratio_val > 2.0:
                # Strong break above R4
                signals[i] = 0.25
                position = 1
            elif close_val < s4_val and vol_ratio_val > 2.0:
                # Strong break below S4
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to pivot or stop at S3
            if close_val < pivot_val or close_val < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to pivot or stop at R3
            if close_val > pivot_val or close_val > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals