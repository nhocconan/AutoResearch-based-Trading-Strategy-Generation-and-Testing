#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Filter_v1
Hypothesis: Use 1d Donchian breakouts with volume confirmation and ATR filter on 12h timeframe.
Donchian breakouts capture strong trends, volume confirms institutional participation,
ATR filter avoids whipsaws in choppy markets. Works in both bull and bear markets
by filtering breakouts with volume and volatility context.
Target: 50-150 total trades over 4 years on 12h timeframe.
"""

name = "12h_Donchian_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D Data for Donchian Channels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period high/low)
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, period20_low)
    
    # === 12h Indicators ===
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stop loss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need at least 20 days of data)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ATR filter: avoid trading when ATR is too low (choppy market)
        atr_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + ATR filter
            if close[i] > donchian_high_aligned[i] and volume_confirm and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume confirmation + ATR filter
            elif close[i] < donchian_low_aligned[i] and volume_confirm and atr_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals