#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR stoploss.
- Long: price breaks above Donchian(20) high + price > 1d EMA50 (uptrend) + volume > 1.5x 20-period avg
- Short: price breaks below Donchian(20) low + price < 1d EMA50 (downtrend) + volume > 1.5x 20-period avg
- Exit: ATR-based trailing stop (close < highest_high_since_entry - 2.5*ATR for long, close > lowest_low_since_entry + 2.5*ATR for short)
- Volume confirmation reduces false breakouts in low-participation moves
- 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe to minimize fee drag
- Works in bull markets via trend continuation and bear markets via filtering out counter-trend breakouts
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
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_high = 0.0  # highest high since entry for long
    entry_low = 0.0   # lowest low since entry for short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 14, 50)  # Need 20 for Donchian/volume, 14 for ATR, 50 for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above Donchian(20) high + uptrend + volume spike
            if volume_spike and close[i] > highest_20[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_high = high[i]
            # Short: break below Donchian(20) low + downtrend + volume spike
            elif volume_spike and close[i] < lowest_20[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_low = low[i]
        elif position == 1:
            # Update highest high since entry
            entry_high = max(entry_high, high[i])
            # Long exit: ATR trailing stop
            if close[i] < entry_high - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            entry_low = min(entry_low, low[i])
            # Short exit: ATR trailing stop
            if close[i] > entry_low + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0