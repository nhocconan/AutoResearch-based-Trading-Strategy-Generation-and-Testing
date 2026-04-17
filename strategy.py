# Strategy: 6h_Camarilla_R3_S3_Breakout_Volume_Trend
# Hypothesis: Breakouts beyond daily Camarilla R3/S3 levels with volume confirmation and 1d EMA200 trend filter.
# Works in bull markets via R3 breakouts, in bear markets via S3 breakdowns. Volume ensures institutional participation.
# Target: 50-150 trades over 4 years on 6h timeframe.

#!/usr/bin/env python3
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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 2
    s3_1d = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get daily EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume and above EMA200
            if close[i] > r3_1d_aligned[i] and volume_filter[i] and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and below EMA200
            elif close[i] < s3_1d_aligned[i] and volume_filter[i] and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or reverses at R4
            r4_1d = close_1d + range_1d * 1.1
            r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
            if close[i] < s3_1d_aligned[i] or close[i] > r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or reverses at S4
            s4_1d = close_1d - range_1d * 1.1
            s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
            if close[i] > r3_1d_aligned[i] or close[i] < s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0