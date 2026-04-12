#!/usr/bin/env python3
"""
6h_1d_1w_Volume_Weighted_Average_Price_VWAP_Breakout
Hypothesis: 6h timeframe with VWAP from 1d and 1w timeframes, using VWAP deviation and volume confirmation.
Trades when price deviates significantly from VWAP with high volume, expecting mean reversion to VWAP.
Works in both bull and bear markets as VWAP acts as dynamic support/resistance.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Volume_Weighted_Average_Price_VWAP_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate VWAP for 1d and 1w
    # VWAP = sum(price * volume) / sum(volume)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w_values = vwap_1w.values
    
    # Align VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w_values)
    
    # Calculate volume average (20 period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Calculate deviation from VWAP (as percentage)
        deviation_1d = (close[i] - vwap_1d_aligned[i]) / vwap_1d_aligned[i]
        deviation_1w = (close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i]
        
        # Entry conditions: significant deviation from VWAP with volume spike
        # Long when price is significantly below VWAP (oversold)
        # Short when price is significantly above VWAP (overbought)
        long_entry = (deviation_1d < -0.015) and (deviation_1w < -0.01) and volume_spike
        short_entry = (deviation_1d > 0.015) and (deviation_1w > 0.01) and volume_spike
        
        # Exit conditions: return to VWAP or opposite deviation
        long_exit = (close[i] > vwap_1d_aligned[i]) or (deviation_1d > -0.005)
        short_exit = (close[i] < vwap_1d_aligned[i]) or (deviation_1d < 0.005)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals