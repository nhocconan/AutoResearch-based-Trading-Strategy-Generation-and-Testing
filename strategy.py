#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Volume Confirmation
Hypothesis: Donchian(20) breakouts on 12h capture trend continuation, 
while 1d volume spikes confirm institutional participation. 
Works in bull (breakouts above upper band) and bear (breakdowns below lower band).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14396_12h_donchian20_1d_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume confirmation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 12h
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # 1d volume confirmation: current 1d volume > 1.5x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    vol_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d.astype(float)).astype(bool)
    
    # ATR for stoploss (12h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = donchian_window + 20  # max(donchian_window, vol_ma window) + buffer
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower band OR stoploss
            if (close[i] <= lower[i] or close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper band OR stoploss
            if (close[i] >= upper[i] or close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume confirmation (both timeframes)
            long_breakout = close[i] > upper[i-1]  # break above previous upper band
            short_breakout = close[i] < lower[i-1]  # break below previous lower band
            
            # Require volume confirmation on both 12h and 1d
            vol_confirm = vol_filter[i] and vol_filter_1d_aligned[i]
            
            if long_breakout and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals