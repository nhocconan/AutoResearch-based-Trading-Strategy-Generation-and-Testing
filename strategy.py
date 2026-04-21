#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ATR Stop
Hypothesis: Donchian channel breakouts (20-period) combined with volume spikes capture 
institutional momentum moves. The strategy works in both bull and bear markets by 
using volatility-based stops and avoiding overtrading through strict volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for ATR and Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily True Range for ATR
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Align daily indicators to 4h timeframe
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    high_20_aligned = align_htf_to_ltf(prices, df_daily, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_daily, low_20)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_daily_aligned[i]) or np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_daily_aligned[i]
        upper_channel = high_20_aligned[i]
        lower_channel = low_20_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.5x 30-period average (strict to reduce trades)
        vol_ma = np.mean(volume[max(0, i-30):i]) if i >= 30 else volume[i]
        vol_ok = vol_current > 2.5 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian band with volume confirmation
            if price > upper_channel and vol_ok:
                signals[i] = 0.30
                position = 1
            # Short breakdown: price breaks below lower Donchian band with volume confirmation
            elif price < lower_channel and vol_ok:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band or ATR-based stop
            if price < lower_channel or (i > 0 and close[i-1] > lower_channel and price < close[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band or ATR-based stop
            if price > upper_channel or (i > 0 and close[i-1] < upper_channel and price > close[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0