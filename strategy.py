#!/usr/bin/env python3
"""
12h_1d_vwap_touch_reversion
Uses 1d VWAP as dynamic support/resistance with volume confirmation.
Long when price touches VWAP from below with volume, short when touches from above.
Exit when price moves away from VWAP or volume dries up.
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
Works in both trending and ranging markets by fading extended moves back to VWAP.
"""

name = "12h_1d_vwap_touch_reversion"
timeframe = "12h"
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate VWAP for 1d: cumulative (price * volume) / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pv = typical_price * df_1d['volume']
    vwap = pv.cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Align VWAP to 12h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # Volume confirmation on 12h: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Price deviation from VWAP for entry sensitivity
    price_dev = (close - vwap_aligned) / vwap_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(vwap_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches VWAP from below with volume (oversold bounce)
        if price_dev[i] <= -0.005 and price_dev[i-1] > -0.005 and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price touches VWAP from above with volume (overbought rejection)
        elif price_dev[i] >= 0.005 and price_dev[i-1] < 0.005 and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price moves away from VWAP or volume dries up
        elif position == 1 and (price_dev[i] >= 0.01 or not vol_confirm[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (price_dev[i] <= -0.01 or not vol_confirm[i]):
            position = 0
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