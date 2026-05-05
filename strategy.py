#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Camarilla pivot levels (R3/S3) from daily timeframe with volume spike confirmation
# Long when price breaks above daily Camarilla R3 level AND volume > 2.0 * 20-period average volume
# Short when price breaks below daily Camarilla S3 level AND volume > 2.0 * 20-period average volume
# Exit when price returns to daily Camarilla pivot point (mean reversion) OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels provide institutional support/resistance that works in both bull and bear markets
# Volume spike confirms breakout strength and reduces false signals
# Pivot point exit provides natural mean reversion target

name = "12h_Camarilla_R3S3_Breakout_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R3 = Pivot + (Range * 1.1 / 2)
    # S3 = Pivot - (Range * 1.1 / 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    s3_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h timeframe (wait for completed daily bar)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily R3 with volume confirmation
            if close[i] > r3_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below daily S3 with volume confirmation
            elif close[i] < s3_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price returns to daily pivot point OR volume drops below average
            if close[i] <= pivot_1d_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price returns to daily pivot point OR volume drops below average
            if close[i] >= pivot_1d_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals