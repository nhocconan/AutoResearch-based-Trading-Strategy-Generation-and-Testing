#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Camarilla pivot levels (R1/S1) and volume confirmation.
# Uses 1d Camarilla pivot levels for mean-reversion entries and exits.
# Only enters when volume > 2x average to avoid false breakouts.
# Targets 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.
# Works in bull/bear by fading extremes at pivot levels with volume confirmation.
name = "12h_1d_Camarilla_R1S1_Volume"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla pivot levels (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_ = high_1d - low_1d
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    r1 = close_1d + range_ * 1.1 / 12.0
    s1 = close_1d - range_ * 1.1 / 12.0
    
    # AlCamarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 2 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below S1 with volume (mean reversion from support)
            if close[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above R1 with volume (mean reversion from resistance)
            elif close[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses above pivot (mean reversion complete)
            if close[i] > ((r1_aligned[i] + s1_aligned[i]) / 2.0):  # midpoint as pivot proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses below pivot (mean reversion complete)
            if close[i] < ((r1_aligned[i] + s1_aligned[i]) / 2.0):  # midpoint as pivot proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals