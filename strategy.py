#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot alignment + volume confirmation
# - Primary signal: Donchian breakout on 6h timeframe (new 20-period high/low)
# - Trend filter: 1d Camarilla pivot levels - only long when price > daily pivot, short when price < daily pivot
# - Volume confirmation: 6h volume > 1.5 * 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, Camarilla pivot provides institutional reference levels
# - Exit: price retrace to midpoint of Donchian channel or opposite breakout

name = "6h_1d_donchian_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels based on previous day
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First value will be invalid due to roll, but min_periods will handle it
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    r4 = pivot + (range_val * 1.1)
    s4 = pivot - (range_val * 1.1)
    
    # Align Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute Donchian channel on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Donchian(20) - highest high and lowest low over 20 periods
    highest_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 6h volume regime: volume > 1.5 * 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retrace to Donchian midpoint OR Donchian breakdown
            if close_6h[i] <= donchian_mid[i] or close_6h[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retrace to Donchian midpoint OR Donchian breakout
            if close_6h[i] >= donchian_mid[i] or close_6h[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with Camarilla alignment and volume confirmation
            # Bullish breakout: price above Donchian upper band AND above daily pivot AND volume
            if (close_6h[i] > highest_high[i] and 
                close_6h[i] > pivot_aligned[i] and 
                volume_regime[i]):
                position = 1
                signals[i] = 0.25
            # Bearish breakout: price below Donchian lower band AND below daily pivot AND volume
            elif (close_6h[i] < lowest_low[i] and 
                  close_6h[i] < pivot_aligned[i] and 
                  volume_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals