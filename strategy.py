#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRFilter_V1
Hypothesis: Donchian(20) breakout with ATR-based stoploss and volume confirmation works on 4h timeframe for BTC and ETH in both bull and bear markets. Uses tight entry conditions (volume > 1.5x 20-bar MA) to limit trades and avoid fee drag. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Volume filter: 20-period average on 4h timeframe
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss and position sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average to reduce trades)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume
            if price > donchian_high[i]:
                if volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian low with volume
            elif price < donchian_low[i]:
                if volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price closes below Donchian low or ATR stoploss
            if price < donchian_low[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Donchian high or ATR stoploss
            if price > donchian_high[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0