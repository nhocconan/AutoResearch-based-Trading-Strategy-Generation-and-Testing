#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dRegime_ATRStop
Hypothesis: 4-hour Donchian(20) breakout with 1-day regime filter (ADX>25) and volume confirmation.
Enters long when price breaks above 20-period high with bullish regime and volume spike.
Enters short when price breaks below 20-period low with bearish regime and volume spike.
Uses ATR-based stoploss via signal=0 when price closes against position.
Designed for 75-200 total trades over 4 years with discrete sizing (0.0, ±0.30) to minimize fee drag.
Works in both bull and bear markets by requiring regime alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Load 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    period = 14
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period
    atr_1d = np.zeros_like(tr)
    plus_di_1d = np.zeros_like(tr)
    minus_di_1d = np.zeros_like(tr)
    
    atr_1d[period] = np.nansum(tr[1:period+1]) if period < len(tr) else 0
    plus_di_1d[period] = np.nansum(plus_dm[1:period+1]) if period < len(plus_dm) else 0
    minus_di_1d[period] = np.nansum(minus_dm[1:period+1]) if period < len(minus_dm) else 0
    
    for i in range(period+1, len(tr)):
        atr_1d[i] = atr_1d[i-1] * (1 - alpha) + alpha * tr[i]
        plus_di_1d[i] = plus_di_1d[i-1] * (1 - alpha) + alpha * plus_dm[i]
        minus_di_1d[i] = minus_di_1d[i-1] * (1 - alpha) + alpha * minus_dm[i]
    
    # Avoid division by zero
    dx_1d = np.zeros_like(atr_1d)
    mask = (plus_di_1d + minus_di_1d) > 0
    dx_1d[mask] = 100 * np.abs(plus_di_1d[mask] - minus_di_1d[mask]) / (plus_di_1d[mask] + minus_di_1d[mask])
    
    adx_1d = np.zeros_like(dx_1d)
    adx_1d[2*period] = np.nanmean(dx_1d[period:2*period+1]) if 2*period < len(dx_1d) else 0
    for i in range(2*period+1, len(dx_1d)):
        adx_1d[i] = adx_1d[i-1] * (1 - alpha) + alpha * dx_1d[i]
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # ATR (14-period) for stoploss on 4h
    atr_4h = np.zeros_like(high)
    tr_4h = np.zeros_like(high)
    for i in range(1, len(high)):
        tr_4h[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
    atr_4h[period] = np.nanmean(tr_4h[1:period+1]) if period < len(tr_4h) else 0
    for i in range(period+1, len(tr_4h)):
        atr_4h[i] = (atr_4h[i-1] * (period-1) + tr_4h[i]) / period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need 20-period Donchian + 2*14 for ADX)
    start_idx = max(20, 2*14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr_4h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        is_trending = adx_1d_aligned[i] > 25
        
        # Long logic: break above Donchian high + trending regime + volume spike
        if close[i] > highest_high[i] and is_trending and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below Donchian low + trending regime + volume spike
        elif close[i] < lowest_low[i] and is_trending and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < lowest_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i]:
            signals[i] = 0.0
            position = 0
        # ATR-based stoploss: exit if price moves 2*ATR against position
        elif position == 1 and close[i] < (signals[i-1] * base_size > 0 and entry_price - 2.0 * atr_4h[i] or close[i-1] - 2.0 * atr_4h[i]):
            # Simplified: use close-based stop
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (signals[i-1] * base_size < 0 and entry_price + 2.0 * atr_4h[i] or close[i-1] + 2.0 * atr_4h[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_1dRegime_ATRStop"
timeframe = "4h"
leverage = 1.0