#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot from 1D + Volume Spike
# Hypothesis: At 6H, fade at Camarilla R3/S3 levels (mean reversion) when price touches these levels with volume spike (>1.5x average).
# Works in ranging markets (reversion to mean) and can capture breakouts when price breaks R4/S4 with volume.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_camarilla_pivot_1d_volume_v1"
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
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Formula: 
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # R2 = C + (H-L) * 1.1/6
    # R1 = C + (H-L) * 1.1/12
    # PP = (H+L+C)/3
    # S1 = C - (H-L) * 1.1/12
    # S2 = C - (H-L) * 1.1/6
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    
    # We need previous day's data to calculate today's levels
    # So we shift the daily data by 1
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivots
    pp = (prev_high + prev_low + prev_close) / 3.0
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 4.0
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 4.0
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align to 6H timeframe (these levels are valid for the entire day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if pivot data not available (first day)
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or breaks above R4 (breakout, reverse)
            if close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_aligned[i]:
                # Breakout above R4 - reverse to short
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or breaks below S4 (breakout, reverse)
            if close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_aligned[i]:
                # Breakdown below S4 - reverse to long
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_spike[i]:
                # Fade at S3 (support) - go long
                if close[i] <= s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Fade at R3 (resistance) - go short
                elif close[i] >= r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                # Breakout above R4 - go long
                elif close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below S4 - go short
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals