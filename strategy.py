#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Long when price breaks above 20-period high with volume > 1.8x 20-period average.
Short when price breaks below 20-period low with volume > 1.8x 20-period average.
Exit via ATR trailing stop (3x ATR from extreme) or opposite Donchian breakout.
Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets by capturing breakouts with volume confirmation.
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
    
    # Calculate ATR(14) for trailing stop
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
    
    # Calculate Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Calculate volume spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_high = 0.0  # for long trailing stop
    lowest_low = 0.0    # for short trailing stop
    
    start_idx = 40  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike
            if price > donch_upper[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_high = price
            # Short: price breaks below Donchian lower with volume spike
            elif price < donch_lower[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_low = price
        
        elif position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, price)
            # ATR trailing stop: exit if price drops 3*ATR from highest high
            if price <= highest_high - 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Also exit on opposite Donchian breakout
            elif price < donch_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, price)
            # ATR trailing stop: exit if price rises 3*ATR from lowest low
            if price >= lowest_low + 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Also exit on opposite Donchian breakout
            elif price > donch_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0