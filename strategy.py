#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ATRFilter_V3
Hypothesis: Donchian channel breakout with volume confirmation and ATR trend filter on 4h timeframe.
Works in bull/bear markets: breakouts capture momentum in trending regimes, volume filter reduces false signals.
ATR filter ensures we only trade in sufficient volatility regimes. Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    upper_channel = high_roll.values
    lower_channel = low_roll.values
    
    # Calculate ATR (14-period) for trend filter and volatility regime
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    volume_ok = prices['volume'].values > (1.3 * vol_ma)
    
    # ATR filter: only trade when ATR > 0.5 * 50-period ATR mean (avoid low volatility)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ok = atr > (0.5 * atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian channel + volume + ATR filter
            if (price > upper_channel[i] and volume_ok[i] and atr_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian channel + volume + ATR filter
            elif (price < lower_channel[i] and volume_ok[i] and atr_ok[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below midpoint of Donchian channel
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above midpoint of Donchian channel
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATRFilter_V3"
timeframe = "4h"
leverage = 1.0