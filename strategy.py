#!/usr/bin/env python3
"""
12h Donchian(10) breakout with 1d EMA(20) trend and volume confirmation
Hypothesis: Price breaking Donchian(10) channels on 12h with 1d EMA(20) trend alignment and volume surge captures strong moves while avoiding whipsaws. Uses tight entry conditions to limit trades and reduce fee drag. Works in bull (long on upper break) and bear (short on lower break).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian10_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA(20) trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(20) for trend direction
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(10) channels - tighter for fewer signals
    highest_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume filter - require significant surge
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # ATR(10) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 30  # For EMA20 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR stoploss
            if (close[i] <= lowest_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR stoploss
            if (close[i] >= highest_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend alignment + volume
            long_breakout = close[i] > highest_high[i-1]  # Break above previous upper
            short_breakout = close[i] < lowest_low[i-1]   # Break below previous lower
            
            uptrend = ema_20_1d_aligned[i] > close[i]  # Price above EMA20
            downtrend = ema_20_1d_aligned[i] < close[i]  # Price below EMA20
            
            if long_breakout and uptrend and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals