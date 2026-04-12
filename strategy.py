#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_frequency_volume_breakout_v1
# Uses 6h price breaking above/below 1-day VWAP bands with volume confirmation.
# In bull markets: buys when price breaks above VWAP(1d) + 1*std with volume surge.
# In bear markets: sells when price breaks below VWAP(1d) - 1*std with volume surge.
# VWAP bands act as dynamic support/resistance; volume confirms institutional interest.
# Target: 15-30 trades/year per symbol to minimize fee drag.
name = "6h_1d_frequency_volume_breakout_v1"
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day VWAP and standard deviation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Calculate standard deviation of price from VWAP
    sq_dev = (typical_price - vwap) ** 2
    var = (sq_dev * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    std_dev = np.sqrt(var.values)
    
    # VWAP bands: VWAP ± 1 standard deviation
    vwap_upper = vwap_values + std_dev
    vwap_lower = vwap_values - std_dev
    
    # Align VWAP bands to 6h timeframe
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if bands not ready
        if np.isnan(vwap_upper_aligned[i]) or np.isnan(vwap_lower_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above VWAP upper band with volume
        if close[i] > vwap_upper_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below VWAP lower band with volume
        elif close[i] < vwap_lower_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns inside VWAP bands
        elif vwap_lower_aligned[i] <= close[i] <= vwap_upper_aligned[i]:
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals