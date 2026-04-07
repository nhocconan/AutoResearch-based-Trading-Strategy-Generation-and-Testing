#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with Volume Confirmation
# Hypothesis: Fade at extreme Camarilla levels (R4/S4) with volume confirmation captures
# mean-reversion in ranging markets, while breakouts beyond R4/S4 with volume capture
# momentum in trending markets. Works in both bull/bear regimes by adapting to price
# action relative to daily pivot structure.
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.

name = "6h_camarilla_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Previous day's values (shift by 1 for proper alignment)
    prev_high = np.concatenate([[np.nan], daily_high[:-1]])
    prev_low = np.concatenate([[np.nan], daily_low[:-1]])
    prev_close = np.concatenate([[np.nan], daily_close[:-1]])
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_val * 1.1 / 2)
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align daily levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (take profit) or breaks below S4 (stop)
            if close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches R3 (take profit) or breaks above R4 (stop)
            if close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Fade at extreme levels with volume
            if (close[i] <= s4_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] >= r4_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Breakout continuation with volume (strong momentum)
            elif (close[i] > r4_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] < s4_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals