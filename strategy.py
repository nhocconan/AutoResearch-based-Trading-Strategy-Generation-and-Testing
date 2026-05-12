#!/usr/bin/env python3
"""
6h Donchian Breakout + Volume + ADX Trend Filter
Hypothesis: Donchian channels capture breakouts in trending markets. Combined with
volume confirmation and ADX > 25 to ensure trend strength, this strategy avoids
false breakouts in ranging markets. Designed for low trade frequency (~20/year)
to minimize fee decay while capturing sustained moves in both bull and bear markets.
"""
name = "6h_Donchian_Breakout_Volume_ADX"
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
    
    # === Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === ADX (14) for trend strength ===
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + ADX > 25 + volume spike
            if (close[i] > highest_high[i] and 
                adx[i] > 25 and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + ADX > 25 + volume spike
            elif (close[i] < lowest_low[i] and 
                  adx[i] > 25 and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR ADX < 20 (trend weakening)
            if close[i] < lowest_low[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR ADX < 20 (trend weakening)
            if close[i] > highest_high[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals