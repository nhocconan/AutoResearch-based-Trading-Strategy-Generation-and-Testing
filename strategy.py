#!/usr/bin/env python3
"""
12h_Donchian_20_200EMA_Volume_Filter
Hypothesis: 12h Donchian(20) breakout with 200EMA trend filter and volume confirmation.
Works in bull markets by buying breakouts above 20-period high when above 200EMA,
and in bear markets by selling breakdowns below 20-period low when below 200EMA.
Volume > 1.5x 20-period average confirms breakout strength. Targets 15-25 trades/year
by requiring all three conditions. Uses 1d ATR for volatility-adjusted position sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels and 200EMA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = np.full_like(high_12h, np.nan)
    low_20 = np.full_like(low_12h, np.nan)
    for i in range(20, len(high_12h)):
        high_20[i] = np.max(high_12h[i-20:i])
        low_20[i] = np.min(low_12h[i-20:i])
    
    # Calculate 12h 200EMA
    close_series = pd.Series(close_12h)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h indicators to lower timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Get 1d data for ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14-period)
    tr1 = np.zeros(len(high_1d))
    tr2 = np.zeros(len(high_1d))
    tr3 = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        tr1[i] = abs(high_1d[i] - low_1d[i])
        tr2[i] = abs(high_1d[i] - close_1d[i-1])
        tr3[i] = abs(low_1d[i] - close_1d[i-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.mean(tr[i-14:i])
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # need Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 20-period high, above 200EMA, with volume
            if (close[i] > high_20_aligned[i] and 
                close[i] > ema_200_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-period low, below 200EMA, with volume
            elif (close[i] < low_20_aligned[i] and 
                  close[i] < ema_200_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price closes below 200EMA or breaks below 20-period low
            if (close[i] < ema_200_aligned[i] or 
                close[i] < low_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 200EMA or breaks above 20-period high
            if (close[i] > ema_200_aligned[i] or 
                close[i] > high_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_200EMA_Volume_Filter"
timeframe = "12h"
leverage = 1.0