#!/usr/bin/env python3
"""
4h Donchian breakout with 12h trend filter and volume confirmation
Hypothesis: In trending markets, breakouts of Donchian channels (20-period) on 4h with 12h trend alignment capture strong directional moves. Volume confirms institutional participation. Works in both bull and bear markets by following the trend.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_trend_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend direction
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require high volume
    
    # 4h ATR(14) for stoploss
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
    start = 50  # For EMA50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: break below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: break above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend alignment + volume
            long_breakout = close[i] > donchian_high[i-1]  # Break above previous high
            short_breakout = close[i] < donchian_low[i-1]  # Break below previous low
            uptrend = ema_12h_aligned[i] > ema_12h_aligned[i-1]  # Rising 12h EMA
            downtrend = ema_12h_aligned[i] < ema_12h_aligned[i-1]  # Falling 12h EMA
            
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