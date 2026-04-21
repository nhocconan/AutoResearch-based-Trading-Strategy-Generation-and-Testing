#!/usr/bin/env python3
"""
12h_HTF_Donchian20_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use 1d Donchian(20) breakout with volume spike (>2x 20-bar MA) and ATR(14) stoploss (2.0x) on 12h timeframe. Donchian channels provide robust trend-following structure, volume confirms breakout strength, ATR stop manages risk. Designed to work in both bull (catch momentum) and bear (fade false breaks via tight stops) markets. Target 12-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for daily Donchian channels
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Donchian Channels (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # === 12h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above 1d Donchian high with volume spike
            if price > donch_high_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Donchian low with volume spike
            elif price < donch_low_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite breakout
            if price < donch_high_aligned[i-1] - 2.0 * atr[i] or price < donch_low_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite breakout
            if price > donch_low_aligned[i-1] + 2.0 * atr[i] or price > donch_high_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_Donchian20_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "12h"
leverage = 1.0