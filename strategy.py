#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with breakout continuation logic
# Fade at R3/S3 (mean reversion) and breakout continuation at R4/S4 (trend following)
# Volume confirmation filters false signals
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: Camarilla pivots adapt to volatility, volume confirms validity

name = "6h_1d_camarilla_breakout_fade_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    hl_range = high_1d - low_1d
    r4 = close_1d + 1.5 * hl_range
    r3 = close_1d + 1.0 * hl_range
    s3 = close_1d - 1.0 * hl_range
    s4 = close_1d - 1.5 * hl_range
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    r4 = np.roll(r4, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    r4[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    s4[0] = np.nan
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute volume confirmation (24-period average for 6d equivalent)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_24[i]
        
        if position == 1:  # Long position
            # Exit long if price crosses below R3 (take profit) or reverses below entry
            if close[i] < r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price crosses above S3 (take profit) or reverses above entry
            if close[i] > s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Camarilla breakout/fade logic with volume confirmation
            if volume_confirmed:
                # Breakout continuation: go long on break above R4, short on break below S4
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                # Fade logic: go short at R3, long at S3 (mean reversion)
                elif close[i] > r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                elif close[i] < s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
    
    return signals