#!/usr/bin/env python3
"""
6h Donchian(20) breakout with weekly trend filter and volume confirmation
Hypothesis: 6h Donchian breakouts capture intermediate trends, filtered by weekly trend direction
to avoid counter-trend trades. Volume confirmation reduces false breakouts. Works in bull (buy
breakouts in uptrend) and bear (sell breakdowns in downtrend).
Target: 75-175 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(50) for trend
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 6h timeframe
    weekly_ema_6h = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Load daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period high/low)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    high_max = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    donch_high_6h = align_htf_to_ltf(prices, df_daily, high_max)
    donch_low_6h = align_htf_to_ltf(prices, df_daily, low_min)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    
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
    start = 100  # For weekly EMA, Donchian and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high_6h[i]) or np.isnan(donch_low_6h[i]) or 
            np.isnan(weekly_ema_6h[i]) or np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_trend_up = weekly_close[-1] > weekly_ema[-1] if i == n-1 else \
                          pd.Series(weekly_close[:i//(24*7)+1]).ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1] > \
                          pd.Series(weekly_close[:i//(24*7)+1]).ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
        # Simplified: use pre-aligned weekly EMA vs price
        weekly_trend_up = close[i] > weekly_ema_6h[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or breakdown below Donchian low
            if (close[i] <= entry_price - 2.5 * atr[i] or
                close[i] <= donch_low_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss or breakout above Donchian high
            if (close[i] >= entry_price + 2.5 * atr[i] or
                close[i] >= donch_high_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume confirmation and trend filter
            breakout_long = (close[i] > donch_high_6h[i] and
                           volume[i] > vol_ema[i] * 1.5 and
                           weekly_trend_up)  # Only long in weekly uptrend
            breakout_short = (close[i] < donch_low_6h[i] and
                            volume[i] > vol_ema[i] * 1.5 and
                            not weekly_trend_up)  # Only short in weekly downtrend
            
            if breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals