#!/usr/bin/env python3
"""
1h 4h/1d Donchian Breakout with Volume and Session Filter
Hypothesis: Use 4h/1d Donchian channels for trend direction, 1h for precise entry with volume confirmation and session filter (08-20 UTC) to reduce noise. Target 60-150 trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian_4h1d_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 4h and 1d HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high_4h = np.full(len(high_4h), np.nan)
    lowest_low_4h = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        highest_high_4h[i] = np.max(high_4h[i-20:i])
        lowest_low_4h[i] = np.min(low_4h[i-20:i])
    donchian_high_4h = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    donchian_low_4h = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Calculate 1d Donchian (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    highest_high_1d = np.full(len(high_1d), np.nan)
    lowest_low_1d = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        highest_high_1d[i] = np.max(high_1d[i-20:i])
        lowest_low_1d[i] = np.min(low_1d[i-20:i])
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below 4h Donchian lower OR 1d Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < donchian_low_4h[i] or
                close[i] < donchian_low_1d[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above 4h Donchian upper OR 1d Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > donchian_high_4h[i] or
                close[i] > donchian_high_1d[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout (4h or 1d) + volume + session
            bull_breakout = close[i] > donchian_high_4h[i] or close[i] > donchian_high_1d[i]
            bear_breakout = close[i] < donchian_low_4h[i] or close[i] < donchian_low_1d[i]
            
            if bull_breakout and volume_filter:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals