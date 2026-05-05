#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# Long when: Price breaks above Donchian upper channel (20) AND 12h EMA(50) > price (uptrend) AND volume > 1.5x 20-period average volume
# Short when: Price breaks below Donchian lower channel (20) AND 12h EMA(50) < price (downtrend) AND volume > 1.5x 20-period average volume
# Exit when price returns to Donchian middle (mean of upper and lower channel)
# Donchian breakout captures sustained momentum after consolidation
# 12h EMA filter ensures we trade in direction of higher timeframe trend
# Volume confirmation adds conviction to breakouts
# Works in both bull and bear markets by filtering breakouts with higher timeframe trend
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_DonchianBreakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA(50)
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_20 + lowest_20) / 2.0
    
    # Calculate 20-period average volume for confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume_20[i]
        
        if position == 0:
            # Long: Break above upper Donchian channel with uptrend and volume
            if close[i] > highest_20[i] and ema_50_12h_aligned[i] > close[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian channel with downtrend and volume
            elif close[i] < lowest_20[i] and ema_50_12h_aligned[i] < close[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to Donchian middle (mean reversion)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to Donchian middle (mean reversion)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals