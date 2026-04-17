#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Long when price breaks above upper Donchian channel with volume > 1.5x average.
Short when price breaks below lower Donchian channel with volume > 1.5x average.
Trailing stop: exit long when price drops 2.5x ATR from highest high since entry.
Exit short when price rises 2.5x ATR from lowest low since entry.
Uses discrete position sizing (0.25) to minimize fee churn. Designed to capture trends in both bull and bear markets while avoiding whipsaws.
"""

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
    
    # Calculate ATR for trailing stop (14-period)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[14] = np.mean(tr[1:15])  # Seed with first 14 values
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Calculate Donchian channels (20-period)
    upper_channel = np.zeros(n)
    lower_channel = np.zeros(n)
    
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-19:i+1])
        lower_channel[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation
            if price > upper and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_high_since_entry = price
            # Short: price breaks below lower Donchian with volume confirmation
            elif price < lower and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, price)
            # Trailing stop: exit if price drops 2.5x ATR from highest high
            if price <= highest_high_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, price)
            # Trailing stop: exit if price rises 2.5x ATR from lowest low
            if price >= lowest_low_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ATRTrail"
timeframe = "4h"
leverage = 1.0