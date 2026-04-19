#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Pivot Point Reversal strategy.
# Uses daily pivot points (PP, R1, S1, R2, S2) to identify reversal zones.
# Enters long when price touches S1/S2 with bullish rejection (close > open).
# Enters short when price touches R1/R2 with bearish rejection (close < open).
# Volume confirmation ensures rejection strength.
# Works in ranging markets and during pullbacks in trending markets.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_1d_PivotReversal_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot point calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points and support/resistance levels
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume filter: volume > 1.3 * 24-period average (4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
            
        # Bullish rejection: close > open (green candle)
        bullish_rejection = close[i] > prices['open'].iloc[i]
        # Bearish rejection: close < open (red candle)
        bearish_rejection = close[i] < prices['open'].iloc[i]
        
        if position == 0:
            # Long when price touches S1/S2 with bullish rejection and volume
            if ((abs(close[i] - s1_aligned[i]) < 0.001 * close[i] or 
                 abs(close[i] - s2_aligned[i]) < 0.001 * close[i]) and
                bullish_rejection and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short when price touches R1/R2 with bearish rejection and volume
            elif ((abs(close[i] - r1_aligned[i]) < 0.001 * close[i] or
                   abs(close[i] - r2_aligned[i]) < 0.001 * close[i]) and
                  bearish_rejection and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price reaches pivot point or shows bearish rejection at R1
            if (close[i] >= pp_aligned[i] or 
                (abs(close[i] - r1_aligned[i]) < 0.001 * close[i] and bearish_rejection)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price reaches pivot point or shows bullish rejection at S1
            if (close[i] <= pp_aligned[i] or 
                (abs(close[i] - s1_aligned[i]) < 0.001 * close[i] and bullish_rejection)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals