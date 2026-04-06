#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and Weekly Trend Filter
Hypothesis: Donchian breakouts capture momentum, volume confirms strength, and weekly trend filter ensures alignment with higher timeframe direction. Works in bull (buy breakouts above weekly trend) and bear (sell breakdowns below weekly trend). Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14448_12h_donchian20_vol_weekly_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    
    # Weekly EMA for trend direction
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
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
    start = max(20, 20) + 1  # Donchian period + buffer
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish OR stoploss
            if (close[i] <= donchian_low[i] or
                weekly_ema_aligned[i] > close[i] or  # Weekly trend bearish
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish OR stoploss
            if (close[i] >= donchian_high[i] or
                weekly_ema_aligned[i] < close[i] or  # Weekly trend bullish
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend alignment
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            # Weekly trend filter: only take longs in bullish weekly trend, shorts in bearish
            weekly_bullish = weekly_ema_aligned[i] < close[i]
            weekly_bearish = weekly_ema_aligned[i] > close[i]
            
            long_setup = long_breakout and vol_filter[i] and weekly_bullish
            short_setup = short_breakout and vol_filter[i] and weekly_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals