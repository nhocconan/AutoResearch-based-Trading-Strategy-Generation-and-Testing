#!/usr/bin/env python3
"""
4h_HTF_Donchian20_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use 4h Donchian channel (20) breakouts confirmed by volume spike (>2x 20-bar MA) with ATR(14) stoploss (2.0x). 
Donchian provides objective trend structure, volume filters weak breakouts, ATR stop manages risk in volatile markets. 
Works in bull (catch breakouts) and bear (fade false breaks via tight stops) by relying on price structure and volume confirmation.
Target: 20-40 trades/year per symbol (<160 total 4h trades over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above Donchian upper with volume spike
            if price > highest_high[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower with volume spike
            elif price < lowest_low[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < highest_high[i-1] - 2.0 * atr[i] or (price < lowest_low[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > lowest_low[i-1] + 2.0 * atr[i] or (price > highest_high[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Donchian20_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0