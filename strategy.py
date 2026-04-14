#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from daily data with volume spike confirmation
# Uses prior day's Camarilla pivot levels (resistance/support) as entry/exit levels
# Volume spike > 2x average confirms breakout/breakdown strength
# Works in bull markets (breakouts above resistance) and bear markets (breakdowns below support)
# Low turnover expected: ~20-40 trades/year per symbol by using daily pivots

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We'll use R3, R2, S2, S3 for entries/exits
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    R4 = close_1d + daily_range * 1.1 / 2
    R3 = close_1d + daily_range * 1.1 / 4
    R2 = close_1d + daily_range * 1.1 / 6
    R1 = close_1d + daily_range * 1.1 / 12
    S1 = close_1d - daily_range * 1.1 / 12
    S2 = close_1d - daily_range * 1.1 / 6
    S3 = close_1d - daily_range * 1.1 / 4
    S4 = close_1d - daily_range * 1.1 / 2
    
    # Use R3, R2, S2, S3 as our key levels
    resistance_levels = np.column_stack([R4, R3, R2, R1])
    support_levels = np.column_stack([S1, S2, S3, S4])
    
    # Align daily levels to 4h timeframe (use previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(R2_aligned[i]) or
            np.isnan(S2_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above R2 with volume confirmation
            if (close[i] > R2_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below S2 with volume confirmation
            elif (close[i] < S2_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below R1 (take profit) or S3 (stop)
            if close[i] < R1_aligned[i] or close[i] < S3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above S1 (take profit) or R3 (stop)
            if close[i] > S1_aligned[i] or close[i] > R3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0