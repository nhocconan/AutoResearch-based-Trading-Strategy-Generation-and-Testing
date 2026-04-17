#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume spike and 12h EMA trend filter.
Long when price breaks above Donchian upper AND volume > 1.8x average AND 12h EMA34 > 12h EMA89 (bullish trend).
Short when price breaks below Donchian lower AND volume > 1.8x average AND 12h EMA34 < 12h EMA89 (bearish trend).
Exit when price reverts to Donchian midpoint OR 12h EMA trend flips.
Uses 4h for price/volume, 12h for EMA trend to reduce whipsaw. Donchian provides structure, volume filters fakeouts, EMA avoids ranging.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    
    # Align Donchian levels to 4h timeframe (already on 4h, but use align for consistency)
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    midpoint_20_aligned = align_htf_to_ltf(prices, df_4h, midpoint_20)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA34 and EMA89 on 12h
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Trend: bullish when EMA34 > EMA89, bearish when EMA34 < EMA89
    ema_trend_bullish = ema_34 > ema_89
    ema_trend_bearish = ema_34 < ema_89
    
    # Align 12h EMA trend to 4h timeframe
    ema_trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_bullish.astype(float))
    ema_trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(midpoint_20_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(ema_trend_bullish_aligned[i]) or np.isnan(ema_trend_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = highest_20_aligned[i]
        lower = lowest_20_aligned[i]
        midpoint = midpoint_20_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        trend_bull = ema_trend_bullish_aligned[i] > 0.5
        trend_bear = ema_trend_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: price > Donchian upper AND volume > 1.8x avg AND 12h EMA bullish
            if price > upper and vol > 1.8 * vol_ma and trend_bull:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND volume > 1.8x avg AND 12h EMA bearish
            elif price < lower and vol > 1.8 * vol_ma and trend_bear:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian midpoint OR 12h EMA turns bearish
            if price < midpoint or not trend_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian midpoint OR 12h EMA turns bullish
            if price > midpoint or not trend_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMA_Trend_Filter"
timeframe = "4h"
leverage = 1.0