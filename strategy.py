# 1d_1W_Pivot_R1S1_Breakout_Volume - Breakout on weekly pivot with volume confirmation
# Uses weekly pivot points (R1/S1) as structural levels with volume filter
# Weekly timeframe provides stability, daily execution reduces lag
# Works in bull (breakouts up) and bear (breakouts down) via symmetry

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1W_Pivot_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly: Calculate pivot points from PREVIOUS week ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use previous week's OHLC for current week's levels
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    
    # Set first week's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    weekly_r1 = (2 * pivot) - prev_low
    weekly_s1 = (2 * pivot) - prev_high
    
    # === Daily: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align weekly levels to daily timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        r1_val = weekly_r1_aligned[i]
        s1_val = weekly_s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume confirmation
            if (close_val > r1_val and vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume confirmation
            elif (close_val < s1_val and vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls back below weekly S1 (mean reversion)
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above weekly R1 (mean reversion)
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals