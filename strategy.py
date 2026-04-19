#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d VWAP as dynamic support/resistance with volume confirmation.
# VWAP acts as a fair value mean; price above VWAP indicates bullish bias, below indicates bearish.
# Uses volume surge (>2x 20-period average) to confirm institutional interest at VWAP touch.
# Works in bull/bear by following intraday trend relative to VWAP.
# Targets 20-40 trades/year (80-160 total over 4 years) with strict entry conditions.
name = "4h_1d_VWAP_Touch_Volume"
timeframe = "4h"
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
    
    # Get 1d data for VWAP calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Typical price and VWAP calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: volume > 2.0 * 20-period average (using 4h volume)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for VWAP calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP with volume surge
            if (close[i] > vwap_1d_aligned[i] and 
                close[i-1] <= vwap_1d_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP with volume surge
            elif (close[i] < vwap_1d_aligned[i] and 
                  close[i-1] >= vwap_1d_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses back below VWAP
            if close[i] < vwap_1d_aligned[i] and close[i-1] >= vwap_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses back above VWAP
            if close[i] > vwap_1d_aligned[i] and close[i-1] <= vwap_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals