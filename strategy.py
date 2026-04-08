#!/usr/bin/env python3
# 4h_1d_volume_breakout_v4
# Hypothesis: Use 1d volume profile to identify high-volume nodes as support/resistance, and enter on 4h breakouts from these zones with volume confirmation. Works in trending markets by riding breakouts and in ranging markets by fading extreme deviations from value area.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years) by requiring volume confirmation and avoiding choppy markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_volume_breakout_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume profile analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume-weighted average price (VWAP) and standard deviation bands
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Calculate volume-weighted standard deviation
    squared_dev = (typical_price - vwap) ** 2
    var = (squared_dev * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    std_dev = np.sqrt(var.values)
    
    # Define value area (VWAP ± 1 standard deviation)
    vwap_upper = vwap_values + std_dev
    vwap_lower = vwap_values - std_dev
    
    # Align VWAP bands to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 2.0
    
    # Trend filter: price position relative to VWAP bands
    price_vwap_ratio = (close - vwap_aligned) / (vwap_upper_aligned - vwap_lower_aligned + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(vwap_aligned[i]) or np.isnan(vwap_upper_aligned[i]) or np.isnan(vwap_lower_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP or breaks below lower band with low volume
            if close[i] <= vwap_aligned[i] or (close[i] < vwap_lower_aligned[i] and not vol_confirm[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to VWAP or breaks above upper band with low volume
            if close[i] >= vwap_aligned[i] or (close[i] > vwap_upper_aligned[i] and not vol_confirm[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above VWAP upper band with volume confirmation
            if close[i] > vwap_upper_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below VWAP lower band with volume confirmation
            elif close[i] < vwap_lower_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals