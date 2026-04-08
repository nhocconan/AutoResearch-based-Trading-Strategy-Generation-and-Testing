#!/usr/bin/env python3
"""
4h_donchian_breakout_volume_v1
Hypothesis: 4h Donchian breakout with volume confirmation and ATR filter works in both bull and bear markets.
- Long: Price breaks above 20-period high + volume > 1.5x average + ATR > 0
- Short: Price breaks below 20-period low + volume > 1.5x average + ATR > 0
- Exit: Opposite Donchian break or ATR-based stop
- Uses 1d trend filter to avoid counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_ma[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    # ATR (14-period) for stop and filter
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i - 13:i + 1])
    
    # 1d trend filter (close above/below 50 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        ema_50_1d[i] = np.mean(close_1d[i - 49:i + 1])  # Simple MA as EMA proxy
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        strong_volume = vol_ratio > 1.5
        atr_filter = atr[i] > 0  # Always true if ATR calculated
        
        if position == 1:  # Long
            # Exit: price breaks below Donchian low or trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above Donchian high or trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: breakout above Donchian high with volume + uptrend
            if (close[i] > highest_high[i] and strong_volume and atr_filter and
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakdown below Donchian low with volume + downtrend
            elif (close[i] < lowest_low[i] and strong_volume and atr_filter and
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals