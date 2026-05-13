#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA50) and volume confirmation.
# Long when price breaks above Donchian upper(20) AND 12h EMA50 rising AND volume > 1.5x average.
# Short when price breaks below Donchian lower(20) AND 12h EMA50 falling AND volume > 1.5x average.
# Exit when price touches Donchian middle (20-period average of upper/lower) OR volume drops below average.
# Uses 4h timeframe for lower frequency, Donchian for structure, 12h EMA for trend, volume for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies.

name = "4h_Donchian20_12hEMA50_Trend_Volume_v2"
timeframe = "4h"
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
    
    # Calculate Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2.0
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above upper Donchian AND 12h EMA50 rising AND volume confirmation
            if (close[i] > highest_20[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Break below lower Donchian AND 12h EMA50 falling AND volume confirmation
            elif (close[i] < lowest_20[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Touch middle Donchian OR volume drops below average
            if close[i] <= middle_20[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Touch middle Donchian OR volume drops below average
            if close[i] >= middle_20[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals