#!/usr/bin/env python3
"""
4h Donchian(20) breakout with volume confirmation and ADX trend filter
Hypothesis: Price breaking Donchian(20) channels with volume surge and ADX>25 trend strength captures momentum moves. Works in bull (long on upper break) and bear (short on lower break). ADX filter reduces false breakouts in ranging markets. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    smoothed_plus_dm = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    smoothed_minus_dm = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * smoothed_plus_dm / atr
    minus_di = 100 * smoothed_minus_dm / atr
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 35  # For ADX and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR ADX weakens (<20) OR stoploss
            if (close[i] <= lowest_low[i] or
                adx[i] < 20 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR ADX weakens (<20) OR stoploss
            if (close[i] >= highest_high[i] or
                adx[i] < 20 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + ADX>25 (trending)
            long_breakout = close[i] > highest_high[i-1]  # Break above previous upper
            short_breakout = close[i] < lowest_low[i-1]   # Break below previous lower
            
            vol_surge = volume[i] > (1.5 * vol_ma[i])  # Volume 1.5x average
            strong_trend = adx[i] > 25  # Strong trend
            
            if long_breakout and vol_surge and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_surge and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals