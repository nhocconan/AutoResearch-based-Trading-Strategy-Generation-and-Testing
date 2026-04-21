#!/usr/bin/env python3
"""
12h Donchian(20) Breakout with Volume Confirmation and ATR Stop
Hypothesis: Donchian channels capture price extremes; breakouts with volume confirm institutional moves.
Uses daily ATR for stops and filters to work in both bull and bear markets. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily True Range for ATR(20)
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align daily ATR to 12h timeframe
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_daily_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_daily_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.8x 30-period average
        vol_ma = np.mean(volume[max(0, i-30):i]) if i >= 30 else volume[i]
        vol_ok = vol_current > 1.8 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume
            if price > upper and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below lower Donchian with volume
            elif price < lower and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or ATR stop
            if price < lower or (i > 0 and close[i-1] > lower and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or ATR stop
            if price > upper or (i > 0 and close[i-1] < upper and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0