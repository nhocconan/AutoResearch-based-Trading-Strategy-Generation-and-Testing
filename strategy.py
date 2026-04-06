#!/usr/bin/env python3
"""
1h Donchian Breakout + Volume Confirmation
Hypothesis: Donchian channel breakouts capture momentum moves. Volume confirms breakout strength.
Use 1d trend filter to avoid counter-trend trades. Works in bull (buy breakouts above channel)
and bear (sell breakdowns below channel) by only trading in direction of 1d trend.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14434_1h_donchian_breakout_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 200, 20)  # Donchian(20), EMA200(200), VolMA(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout + volume + 1d trend filter
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            # 1d trend filter: only long in uptrend, only short in downtrend
            uptrend = close[i] > ema200_1d_aligned[i]
            downtrend = close[i] < ema200_1d_aligned[i]
            
            if long_breakout and vol_filter[i] and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals