# 6h_1w_Pivot_R3S3_Fade_With_Volume_Confirmation
# Hypothesis: Fade at weekly R3/S3 levels with volume confirmation on 6h timeframe.
# Weekly pivot levels act as strong support/resistance; price often reverses from these levels.
# Volume confirmation ensures we only take trades with institutional interest.
# Works in both bull and bear markets as it's a mean-reversion strategy at key levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Pivot_R3S3_Fade_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Calculate weekly R3, S3 levels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Weekly R3 and S3 levels
    r3_1w = pivot_1w + (range_1w * 1.1)
    s3_1w = pivot_1w - (range_1w * 1.1)
    
    # Align weekly levels to 6h
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r3_1w_val = r3_1w_aligned[i]
        s3_1w_val = s3_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_1w_val) or np.isnan(s3_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or goes below S3 with volume confirmation
            if close_val <= s3_1w_val and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above R3 with volume confirmation
            elif close_val >= r3_1w_val and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or above pivot
            if close_val >= pivot_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or below pivot
            if close_val <= pivot_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals