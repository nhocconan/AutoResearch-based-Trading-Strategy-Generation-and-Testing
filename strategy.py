#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h HMA(21) Trend + Volume Spike (2.0x) + ATR Trailing Stop (2.5x)
Hypothesis: Donchian breakouts capture momentum swings. 12h HMA trend filter ensures alignment with higher timeframe direction, reducing whipsaw. Volume spike confirms participation. ATR trailing stop manages risk. Discrete position sizing (0.30) targets ~30-60 trades/year on 4h to minimize fee drag. Works in bull/bear via trend filter and trailing stop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h HMA(21) trend filter (MTF) - loaded ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).mean().values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 21) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr_14[i]) or np.isnan(hma_21_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions: price breaks Donchian(20) channels
        breakout_long = curr_close > highest_20[i]
        breakout_short = curr_close < lowest_20[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + 12h HMA trend alignment
            long_entry = breakout_long and vol_spike and (hma_21_12h_aligned[i] > 0)
            short_entry = breakout_short and vol_spike and (hma_21_12h_aligned[i] < 0)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.30
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hHMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0