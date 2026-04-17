#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Long when price breaks above 20-period high with volume > 1.5x average.
Short when price breaks below 20-period low with volume > 1.5x average.
Exit via ATR trailing stop (3x ATR) or opposite Donchian breakout.
Uses discrete position sizing (0.25) to limit fee drag. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(close)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_channel, lower_channel = donchian_channels(high, low, 20)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if ATR not ready
        if np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        vol_conf = volume_confirm[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume confirmation
            if price > upper and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below lower channel with volume confirmation
            elif price < lower and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 3*ATR from high
            if price < highest_since_entry - 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Opposite breakout exit
            elif price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 3*ATR from low
            if price > lowest_since_entry + 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Opposite breakout exit
            elif price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0