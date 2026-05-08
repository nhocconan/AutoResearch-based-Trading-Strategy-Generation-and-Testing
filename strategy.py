#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour weekly Donchian breakout with daily volume confirmation
# We go long when price breaks above the weekly Donchian high (20 periods) with daily volume spike.
# We go short when price breaks below the weekly Donchian low (20 periods) with daily volume spike.
# Uses 12h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Weekly Donchian channels provide robust trend-following structure in both bull and bear markets.
# Volume spike confirms institutional participation in the breakout.
# Exit occurs when price returns to the weekly Donchian midpoint.

name = "12h_WeeklyDonchian_Breakout_DailyVolume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly upper band (20-period high)
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    # Weekly lower band (20-period low)
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    # Weekly midpoint (for exit)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align weekly Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Daily volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_level = donchian_high_aligned[i]
        low_level = donchian_low_aligned[i]
        mid_level = donchian_mid_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + volume spike
            if close[i] > high_level and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + volume spike
            elif close[i] < low_level and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly Donchian midpoint
            if close[i] < mid_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly Donchian midpoint
            if close[i] > mid_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals