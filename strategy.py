# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h_1d_vwap_mean_reversion
- Uses 1-day VWAP as dynamic mean with 6h price reversion to VWAP
- In bull markets: buy when price dips below VWAP with bullish momentum (close > open)
- In bear markets: sell when price rises above VWAP with bearish momentum (close < open)
- Volume-weighted average price acts as institutional reference point
- Target: 20-40 trades/year per symbol with mean reversion edge
"""
name = "6h_1d_vwap_mean_reversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate typical price and VWAP for 1d
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Align VWAP to 6h timeframe (1-day delay due to calculation needing close)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # 6h momentum: close > open for bullish, close < open for bearish
    bullish_momentum = close > open_price
    bearish_momentum = close < open_price
    
    # Distance from VWAP as percentage
    vwap_distance = (close - vwap_aligned) / vwap_aligned
    
    # Mean reversion thresholds: enter when price deviates significantly from VWAP
    entry_threshold = 0.008  # 0.8% deviation
    exit_threshold = 0.002   # 0.2% deviation for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # start after first bar for momentum calculation
        # Skip if VWAP not ready
        if np.isnan(vwap_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Exit conditions: price returns near VWAP
        if abs(vwap_distance[i]) < exit_threshold:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions with momentum confirmation
        if position == 0:
            # Long: price below VWAP with bullish momentum
            if vwap_distance[i] < -entry_threshold and bullish_momentum[i]:
                position = 1
                signals[i] = 0.25
            # Short: price above VWAP with bearish momentum
            elif vwap_distance[i] > entry_threshold and bearish_momentum[i]:
                position = -1
                signals[i] = -0.25
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