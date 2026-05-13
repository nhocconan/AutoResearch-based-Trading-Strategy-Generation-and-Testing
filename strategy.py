#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
# Long when price breaks above 12h Donchian upper channel AND 1d EMA50 rising AND volume > 1.5x average.
# Short when price breaks below 12h Donchian lower channel AND 1d EMA50 falling AND volume > 1.5x average.
# Exit when price reverts to 12h Donchian midpoint OR trend reversal.
# Uses 12h timeframe for lower frequency, Donchian for structure, 1d EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakouts, bear via faded rallies.

name = "12h_Donchian20_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian(20) on 12h high/low
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    mid_12h = (upper_12h + lower_12h) / 2.0
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA50 slope for trend direction (rising/falling)
    ema50_slope = np.diff(ema50_1d_aligned, prepend=ema50_1d_aligned[0])
    ema50_rising = ema50_slope > 0
    ema50_falling = ema50_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or np.isnan(mid_12h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian AND 1d EMA50 rising AND volume confirmation
            if close[i] > upper_12h[i] and ema50_rising[i] and volume_filter_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian AND 1d EMA50 falling AND volume confirmation
            elif close[i] < lower_12h[i] and ema50_falling[i] and volume_filter_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to midpoint OR trend reversal (EMA50 falling)
            if close[i] < mid_12h[i] or ema50_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to midpoint OR trend reversal (EMA50 rising)
            if close[i] > mid_12h[i] or ema50_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals