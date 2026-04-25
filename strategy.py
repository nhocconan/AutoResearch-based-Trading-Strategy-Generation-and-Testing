#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_ATRVolSpike
Hypothesis: Trade 6h timeframe using Donchian channel (20-bar) breakout in direction of 1d EMA50 trend, 
confirmed by ATR expansion (>1.5x 20-bar ATR MA) and volume spike (>2.0x 20-bar volume MA). 
Enter long when price breaks above upper Donchian AND above 1d EMA50 AND ATR expansion AND volume spike. 
Enter short when price breaks below lower Donchian AND below 1d EMA50 AND ATR expansion AND volume spike. 
Exit on opposite Donchian touch or trend reversal. Uses discrete sizing 0.25. 
Target: 12-30 trades/year on 6h timeframe. Donchian provides objective breakout levels, 
1d EMA50 ensures trend alignment, ATR expansion confirms momentum, volume spike confirms participation. 
Works in bull/bear by only trading with daily trend, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-bar Donchian channels on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14) on 6h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-bar ATR MA for expansion filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_expansion = atr_14 > (1.5 * atr_ma_20)
    
    # Calculate 20-bar volume MA for volume spike
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA50 (50) and 6h indicators (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_ma_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND above 1d EMA50 AND ATR expansion AND volume spike
            long_setup = (close[i] > highest_20[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         atr_expansion[i] and \
                         volume_spike[i]
            # Short: price breaks below lower Donchian AND below 1d EMA50 AND ATR expansion AND volume spike
            short_setup = (close[i] < lowest_20[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          atr_expansion[i] and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches lower Donchian OR closes below 1d EMA50
            if (close[i] <= lowest_20[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian OR closes above 1d EMA50
            if (close[i] >= highest_20[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_ATRVolSpike"
timeframe = "6h"
leverage = 1.0