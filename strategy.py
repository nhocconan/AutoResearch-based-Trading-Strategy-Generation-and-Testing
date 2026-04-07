#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal + Volume Spike
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on 6h timeframe.
# Price rejection at these levels with volume confirmation provides high-probability reversal entries.
# Works in both bull and bear markets by fading extremes and catching mean reversion.
# Target: 20-30 trades/year to minimize fee drag on 6h timeframe.
name = "6h_camarilla_pivot_reversal_v1"
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
    
    # Daily high/low/close for Camarilla calculation (using 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 6h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels
    R3 = daily_close_aligned + 1.1 * (daily_high_aligned - daily_low_aligned) / 6
    S3 = daily_close_aligned - 1.1 * (daily_high_aligned - daily_low_aligned) / 6
    R4 = daily_close_aligned + 1.1 * (daily_high_aligned - daily_low_aligned) / 2
    S4 = daily_close_aligned - 1.1 * (daily_high_aligned - daily_low_aligned) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(R4[i]) or np.isnan(S4[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price rejects S3/S4 with volume spike
        if (low[i] <= S3[i] and close[i] > S3[i]) or (low[i] <= S4[i] and close[i] > S4[i]):
            if vol_spike[i]:
                signals[i] = 0.25  # Long 25%
        
        # Short setup: price rejects R3/R4 with volume spike
        elif (high[i] >= R3[i] and close[i] < R3[i]) or (high[i] >= R4[i] and close[i] < R4[i]):
            if vol_spike[i]:
                signals[i] = -0.25  # Short 25%
    
    return signals