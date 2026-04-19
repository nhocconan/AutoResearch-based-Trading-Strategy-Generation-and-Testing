#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h VWAP as dynamic support/resistance with volume confirmation.
# Enters when price crosses VWAP with volume > 2x average, exits on VWAP reversion.
# VWAP acts as institutional reference point; breaks indicate institutional flow.
# Volume filter ensures conviction. Designed for 20-40 trades/year to avoid fee drag.
# Works in bull (breakouts hold) and bear (fades at VWAP) via mean-reversion exit.
name = "4h_12h_VWAP_Break_Volume"
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
    
    # Get 12h data for VWAP calculation (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Typical price for VWAP
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    vwap_12h = np.cumsum(typical_price_12h * volume_12h) / np.cumsum(volume_12h)
    # Handle division by zero at start
    vwap_12h = np.where(np.cumsum(volume_12h) == 0, typical_price_12h, vwap_12h)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for VWAP and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(vwap_12h_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP with volume confirmation
            if close[i] > vwap_12h_aligned[i] and close[i-1] <= vwap_12h_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP with volume confirmation
            elif close[i] < vwap_12h_aligned[i] and close[i-1] >= vwap_12h_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses back below VWAP
            if close[i] < vwap_12h_aligned[i] and close[i-1] >= vwap_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses back above VWAP
            if close[i] > vwap_12h_aligned[i] and close[i-1] <= vwap_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals