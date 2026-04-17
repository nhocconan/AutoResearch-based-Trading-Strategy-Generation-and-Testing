#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour timeframe with weekly pivot (R1/S1) breakout and 1-day volume confirmation.
# Weekly pivots provide strong institutional support/resistance levels. Breakout above R1 or below S1
# with elevated daily volume indicates institutional participation. Works in bull markets (breakouts continue)
# and bear markets (breakdowns accelerate). Uses 1-day volume to avoid false breakouts on low volume.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for pivot points ===
    df_1w = get_htf_data(prices, '1w')
    # Typical price for pivot calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Calculate weekly pivot points
    pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    r1 = 2 * pivot - df_1w['low']
    s1 = 2 * pivot - df_1w['high']
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    
    # === Daily data for volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # 20-day average volume
    volume_1d_series = pd.Series(volume_1d)
    vol_avg20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    # Current day volume aligned
    vol_1d_current_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current day volume > 1.5x 20-day average
        vol_filter = vol_1d_current_aligned[i] > 1.5 * vol_avg20_1d_aligned[i]
        
        # Long signal: close breaks above R1 with volume confirmation
        if close[i] > r1_aligned[i] and vol_filter:
            if position <= 0:  # Only enter if not already long
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25  # Maintain position
        # Short signal: close breaks below S1 with volume confirmation
        elif close[i] < s1_aligned[i] and vol_filter:
            if position >= 0:  # Only enter if not already short
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25  # Maintain position
        # Exit conditions: return to pivot zone or volume drops
        elif position == 1 and (close[i] < pivot_aligned[i] or not vol_filter):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > pivot_aligned[i] or not vol_filter):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_WeeklyPivot_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0