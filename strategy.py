#!/usr/bin/env python3
"""
6h Weekly Pivot + Donchian(20) Breakout + Volume Spike
Hypothesis: Weekly pivot levels (from prior week) act as key support/resistance on 6h timeframe.
Breakouts above weekly R1 or below weekly S1 with Donchian(20) confirmation and volume spike
indicate institutional participation. Works in bull markets via breakouts with momentum and
in bear markets via fade at extreme weekly levels (R2/S2) when price rejects after spike.
Target: 12-30 trades/year on 6h (50-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week OHLC
    # Standard floor pivot: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    prev_high = df_1w['high'].values
    prev_low = df_1w['low'].values
    prev_close = df_1w['close'].values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rang = prev_high - prev_low
    R1 = 2 * pivot - prev_low
    S1 = 2 * pivot - prev_high
    R2 = pivot + rang
    S2 = pivot - rang
    
    # Align weekly pivot levels to 6h (use previous week's levels for current week's trading)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Calculate Donchian(20) on 6h for breakout confirmation
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        R1_level = R1_aligned[i]
        S1_level = S1_aligned[i]
        R2_level = R2_aligned[i]
        S2_level = S2_aligned[i]
        donchian_high = high_roll[i]
        donchian_low = low_roll[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above weekly R1 AND Donchian(20) high AND volume spike
            long_entry = (curr_close > R1_level) and (curr_high > donchian_high) and vol_spike
            # Short: price breaks below weekly S1 AND Donchian(20) low AND volume spike
            short_entry = (curr_close < S1_level) and (curr_low < donchian_low) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below weekly S1 (reversal to downside) OR
            #        price rejects at weekly R2 with volume spike (fade at resistance)
            if (curr_close < S1_level) or \
               ((curr_close > R2_level) and vol_spike and (curr_high < R2_level * 1.005)):  # rejection at R2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly R1 (reversal to upside) OR
            #        price rejects at weekly S2 with volume spike (fade at support)
            if (curr_close > R1_level) or \
               ((curr_close < S2_level) and vol_spike and (curr_low > S2_level * 0.995)):  # rejection at S2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0