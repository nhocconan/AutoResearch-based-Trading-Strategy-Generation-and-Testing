#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d ATR ratio > 0.7 (low volatility regime) AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND 1d ATR ratio > 0.7 AND volume > 1.5x 20-period average.
Exit when price reverts to Donchian midpoint OR ATR trailing stop (2.5*ATR from extreme).
Uses 1d HTF for volatility regime alignment (ATR ratio = ATR(14)/ATR(50)).
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
Works in both bull and bear markets: volatility regime filter avoids choppy conditions, Donchian breakouts capture sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for ATR(50)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR ratio = ATR(14)/ATR(50) - measures relative volatility (low when ratio < 0.7)
    atr_ratio_1d = np.where(atr_50_1d > 0, atr_14_1d / atr_50_1d, 1.0)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Donchian(20) channels from 12h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # 12h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 12h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 60)  # vol_ma20, donchian20, and ATR(50) needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_ratio = atr_ratio_1d_aligned[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        mid = donchian_mid[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper band AND low volatility regime (ATR ratio < 0.7) AND volume spike
            if price > upper and atr_ratio < 0.7 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below lower band AND low volatility regime AND volume spike
            elif price < lower and atr_ratio < 0.7 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to Donchian midpoint
            if position == 1 and price < mid:
                exit_signal = True
            elif position == -1 and price > mid:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dATR_Ratio_VolumeSpike_MidExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0