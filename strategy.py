#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels + volume confirmation + 1d trend filter
# Uses Camarilla pivot levels from daily data for precise entry/exit points,
# volume to confirm institutional participation, and 1d EMA for trend filter.
# Works in both bull and bear by taking long entries above pivot resistance
# in uptrends and short entries below pivot support in downtrends.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot point = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    # R4 = C + (H - L) * 1.1 / 2
    # S1 = C - (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # S3 = C - (H - L) * 1.1 / 4
    # S4 = C - (H - L) * 1.1 / 2
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    r2_1d = close_1d + (high_1d - low_1d) * 1.1 / 6
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    s2_1d = close_1d - (high_1d - low_1d) * 1.1 / 6
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate EMA25 on 1d for trend filter
    ema25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Calculate volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema25_12h = align_htf_to_ltf(prices, df_1d, ema25_1d)
    vol_avg_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(r1_12h[i]) or
            np.isnan(s1_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(ema25_12h[i]) or np.isnan(vol_avg_12h[i])):
            continue
        
        # Long entry: price breaks above R3 + volume spike + price above EMA25 (uptrend)
        if (close[i] > r3_12h[i] and
            volume[i] > 1.5 * vol_avg_12h[i] and
            close[i] > ema25_12h[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below S3 + volume spike + price below EMA25 (downtrend)
        elif (close[i] < s3_12h[i] and
              volume[i] > 1.5 * vol_avg_12h[i] and
              close[i] < ema25_12h[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to pivot point
        elif position == 1 and close[i] < pp_12h[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pp_12h[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Volume_Trend"
timeframe = "12h"
leverage = 1.0