#!/usr/bin/env python3
"""
12h_HTF_1d_Donchian20_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use 1d HTF Donchian(20) breakout direction + 12h volume spike (>2x 20-bar MA) for entry + ATR(14) stoploss (2.0x). Donchian channels provide robust trend structure, volume spike confirms momentum, ATR stop manages risk. Designed to work in both bull (catch breakouts) and bear (fade false breaks via tight stops) markets. Target 12-30 trades/year per symbol.
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
    
    if len(df_1d) < 21:  # need 20 for Donchian + 1 for previous
        return np.zeros(n)
    
    # === 1d Donchian Channels (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: 20-period high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) 
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
            if price > donchian_high_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Donchian low with volume spike
            elif price < donchian_low_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite breakout
            if price < donchian_high_aligned[i-1] - 2.0 * atr[i] or price < donchian_low_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite breakout
            if price > donchian_low_aligned[i-1] + 2.0 * atr[i] or price > donchian_high_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1d_Donchian20_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "12h"
leverage = 1.0