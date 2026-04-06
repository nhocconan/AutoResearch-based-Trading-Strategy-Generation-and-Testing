#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly trend filter and volume confirmation.
# Breakout above 20-day high (long) or below 20-day low (short) when weekly trend aligns.
# Weekly trend: price above/below 20-week EMA. Volume must be above 20-day average.
# Works in bull markets via breakout continuation and in bear via mean reversion at bands.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_donchian20_weekly_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-day Donchian channels (using previous day's data to avoid look-ahead)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).shift(1).values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).shift(1).values
    
    # Weekly trend filter: 20-week EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema_20w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Volume filter: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price reaches 20-day low (mean reversion) or weekly trend turns bearish
            if close[i] <= low_20[i] or close[i] < ema_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches 20-day high (mean reversion) or weekly trend turns bullish
            if close[i] >= high_20[i] or close[i] > ema_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Breakout long: price above 20-day high AND weekly bullish
                if close[i] > high_20[i] and close[i] > ema_20w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Breakout short: price below 20-day low AND weekly bearish
                elif close[i] < low_20[i] and close[i] < ema_20w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals