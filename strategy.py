#!/usr/bin/env python3
"""
1d Donchian Breakout with Volume and 1w Trend Filter
Hypothesis: On daily timeframe, price breaking Donchian(20) channels with volume confirmation and aligned weekly trend captures strong moves while avoiding false breakouts. Works in bull (long with uptrend) and bear (short with downtrend). Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
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
    start = 50  # For Donchian20 and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks lower Donchian OR stoploss
            if (close[i] <= low_20[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks upper Donchian OR stoploss
            if (close[i] >= high_20[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend alignment
            long_breakout = close[i] > high_20[i-1]  # Break above previous high
            short_breakout = close[i] < low_20[i-1]  # Break below previous low
            volume_confirm = volume[i] > (1.5 * vol_ma[i])  # Above average volume
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]  # Rising weekly EMA
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]  # Falling weekly EMA
            
            if long_breakout and volume_confirm and weekly_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and volume_confirm and weekly_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals