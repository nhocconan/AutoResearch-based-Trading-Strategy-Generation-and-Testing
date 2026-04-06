#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and Trend Filter
Hypothesis: Breakouts of Donchian(20) channels on 4h, confirmed by volume spike and aligned with 1d trend, capture significant moves. Works in bull markets (long breakouts above upper band with uptrend) and bear markets (short breakouts below lower band with downtrend). Volume ensures institutional participation, reducing false breakouts. Target: 100-150 total trades over 4 years (25-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "4h_donchian20_1d_trend_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian band OR stoploss
            if (close[i] <= donch_low[i] or
                close[i] <= entry_price - 2.5 * (donch_high[i-20] - donch_low[i-20]) / 2):  # Approximate ATR
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian band OR stoploss
            if (close[i] >= donch_high[i] or
                close[i] >= entry_price + 2.5 * (donch_high[i-20] - donch_low[i-20]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + volume + trend alignment
            long_breakout = close[i] > donch_high[i-1]  # Break above previous upper band
            short_breakout = close[i] < donch_low[i-1]  # Break below previous lower band
            
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            if long_breakout and vol_filter[i] and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals