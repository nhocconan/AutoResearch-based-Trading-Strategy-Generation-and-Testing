#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d pivot-based mean reversion at extreme levels (R4/S4).
# Uses daily Camarilla pivot levels: fade at R4/S4 with volume confirmation.
# Works in both bull/bear by capturing overextended moves that revert to mean.
# Targets 15-35 trades/year (60-140 total over 4 years) with strict entry conditions.
# Uses 1d Camarilla pivots calculated from prior day's OHLC.
name = "6h_1d_Camarilla_R4S4_Fade_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = C + ((H-L) * 1.1/2)
    # S4 = C - ((H-L) * 1.1/2)
    # where C = close, H = high, L = low of previous day
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align to 6h timeframe (wait for daily bar to close)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches or goes below S4 with volume (mean reversion long)
            if low[i] <= s4_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R4 with volume (mean reversion short)
            elif high[i] >= r4_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to midpoint (mean reversion complete)
            # Midpoint between S4 and R4 is close_1d (the pivot point)
            # We approximate using the 1d close aligned
            if close[i] >= r4_1d_aligned[i] * 0.5 + s4_1d_aligned[i] * 0.5:  # midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to midpoint
            if close[i] <= r4_1d_aligned[i] * 0.5 + s4_1d_aligned[i] * 0.5:  # midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals