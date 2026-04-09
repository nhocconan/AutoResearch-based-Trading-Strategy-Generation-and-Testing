#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy combining 1w Camarilla pivot levels with 1w volume spike filter
# Camarilla pivot levels provide structured support/resistance based on prior day's range
# Volume spike (>2x 20-period average) confirms institutional interest at these levels
# Long when price closes above Camarilla H3 with volume spike, short when below L3 with volume spike
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: breaks above H3 indicate strength, breaks below L3 indicate weakness

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for Camarilla calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 20-period average volume on 1w timeframe
    vol_s_1w = pd.Series(volume_1w)
    avg_vol_20_1w = vol_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Calculate volume spike (>2x 20-period average)
    volume_spike_1w = np.where(volume_1w > 2.0 * avg_vol_20_1w, 1.0, 0.0)
    
    # Calculate Camarilla pivot levels for each 1d bar using prior day's OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low), etc.
    # L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    camarilla_h3_1d = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_l3_1d = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align 1w indicators to 1d timeframe
    volume_spike_1d = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_spike_1d[i]) or np.isnan(camarilla_h3_1d_aligned[i]) or 
            np.isnan(camarilla_l3_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price closes below H3 or volume spike ends
            if close[i] < camarilla_h3_1d_aligned[i] or volume_spike_1d[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price closes above L3 or volume spike ends
            if close[i] > camarilla_l3_1d_aligned[i] or volume_spike_1d[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when price closes above H3 with volume spike
            if close[i] > camarilla_h3_1d_aligned[i] and volume_spike_1d[i] > 0.5:
                position = 1
                signals[i] = 0.25
            # Enter short when price closes below L3 with volume spike
            elif close[i] < camarilla_l3_1d_aligned[i] and volume_spike_1d[i] > 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals