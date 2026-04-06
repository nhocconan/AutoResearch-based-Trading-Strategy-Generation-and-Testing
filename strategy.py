#!/usr/bin/env python3
"""
1d Donchian Breakout + Volume Confirmation + ATR Stop
Hypothesis: Daily Donchian(20) breakouts with volume confirmation and ATR stops capture momentum
in both bull and bear markets. Works as breakout strategy in trends and mean-reversion in ranges
via stop-and-reverse logic. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14410_1d_donchian20_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    weekly_ema = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-day)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # Require 20% above average volume
    
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
    start = max(lookback, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(weekly_ema_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
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
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend filter
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            # Weekly trend filter: only take longs in uptrend, shorts in downtrend
            weekly_uptrend = close[i] > weekly_ema_aligned[i]
            weekly_downtrend = close[i] < weekly_ema_aligned[i]
            
            if long_breakout and vol_filter[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals