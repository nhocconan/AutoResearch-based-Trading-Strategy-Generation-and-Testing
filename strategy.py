#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price position relative to 12h VWAP with 1d volume confirmation
# Price above/below VWAP indicates institutional bias. Volume surge confirms participation.
# This mean-reversion strategy fades extreme deviations from VWAP during high volume.
# Works in ranging markets (common in 2025) and avoids strong trends via volume filter.
# Targets 15-30 trades per year (~60-120 total over 4 years) to minimize fee drag.

name = "6h_VWAPDeviation_1dVolume"
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
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h VWAP (typical price * volume) / volume
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    vwap_numerator = (typical_price * df_12h['volume']).cumsum()
    vwap_denominator = df_12h['volume'].cumsum()
    vwap_12h = vwap_numerator / vwap_denominator
    vwap_12h = vwap_12h.values  # Convert to numpy array
    
    # Align 12h VWAP to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Get 1d data for volume spike detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d volume spike detection (20-period MA)
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_spike_1d = df_1d['volume'].values > (vol_ma.values * 2.0)
    vol_spike = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate deviation from VWAP as percentage
        deviation = (close[i] - vwap_aligned[i]) / vwap_aligned[i]
        
        if position == 0:
            # Enter long when price is significantly below VWAP with volume spike
            if deviation < -0.015 and vol_spike[i]:  # -1.5% deviation
                signals[i] = 0.25
                position = 1
            # Enter short when price is significantly above VWAP with volume spike
            elif deviation > 0.015 and vol_spike[i]:  # +1.5% deviation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price returns to VWAP or deviation reverses
            if deviation > -0.005:  # Close to or above VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price returns to VWAP or deviation reverses
            if deviation < 0.005:  # Close to or below VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals