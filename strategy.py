#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Trend and Volume Filter
Hypothesis: In trending markets, price breaking above/below weekly Donchian channels on 6h timeframe with volume confirmation captures momentum while avoiding false breakouts. Uses weekly trend (price above/below weekly EMA200) to filter direction. Works in bull (long with weekly uptrend) and bear (short with weekly downtrend).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_trend_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    # Load weekly data for trend and Donchian channels (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(200) for long-term trend
    close_weekly = df_weekly['close'].values
    ema200_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False).mean().values
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR(14) for stoploss
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
    start = 200  # For weekly EMA200 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema200_weekly_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below weekly Donchian low OR stoploss
            if (close[i] <= donchian_low_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above weekly Donchian high OR stoploss
            if (close[i] >= donchian_high_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with weekly trend and volume
            long_breakout = (close[i] > donchian_high_aligned[i] and
                            close[i] > ema200_weekly_aligned[i] and
                            volume[i] > 1.5 * vol_ma[i])
            short_breakout = (close[i] < donchian_low_aligned[i] and
                             close[i] < ema200_weekly_aligned[i] and
                             volume[i] > 1.5 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals