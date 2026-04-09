#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
# Donchian breakouts capture momentum; 12h Camarilla provides institutional support/resistance
# Volume confirmation ensures breakout authenticity with conviction
# Works in bull/bear: Camarilla adapts to higher timeframe structure
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_12h_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, len(df_12h)):
        # Previous 12h bar
        phigh = df_12h['high'].iloc[i-1]
        plow = df_12h['low'].iloc[i-1]
        pclose = df_12h['close'].iloc[i-1]
        
        # Pivot point
        pivot = (phigh + plow + pclose) / 3.0
        range_val = phigh - plow
        
        # Camarilla levels
        camarilla_high[i-1] = phigh
        camarilla_low[i-1] = plow
        camarilla_r3[i-1] = pclose + range_val * 1.1 / 4.0
        camarilla_s3[i-1] = pclose - range_val * 1.1 / 4.0
        camarilla_r4[i-1] = pclose + range_val * 1.1 / 2.0
        camarilla_s4[i-1] = pclose - range_val * 1.1 / 2.0
    
    # Align Camarilla data to 6h timeframe (wait for 12h bar close)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high.values)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low.values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3.values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4.values)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < Camarilla S3 (support broken)
            if close[i] < donchian_low[i] or close[i] < camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > Camarilla R3 (resistance broken)
            if close[i] > donchian_high[i] or close[i] > camarilla_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + Camarilla filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND price > Camarilla R3 (breakout above resistance)
                if close[i] > donchian_high[i] and close[i] > camarilla_r3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < Camarilla S3 (breakdown below support)
                elif close[i] < donchian_low[i] and close[i] < camarilla_s3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals