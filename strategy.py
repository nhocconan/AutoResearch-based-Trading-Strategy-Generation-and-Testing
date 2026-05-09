#!/usr/bin/env python3
# Hypothesis: 1d Donchian channel breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high, weekly EMA200 is rising, and volume > 1.5x 20-day average
# Short when price breaks below 20-day low, weekly EMA200 is falling, and volume > 1.5x 20-day average
# Exit when price returns to the 20-day midpoint or weekly trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown
# Designed to capture trends in both bull and bear markets with weekly trend alignment

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # Weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Weekly EMA200 slope (rising/falling)
    ema200_slope = np.diff(ema200_1w_aligned, prepend=ema200_1w_aligned[0])
    ema_rising = ema200_slope > 0
    ema_falling = ema200_slope < 0
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_20[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-day high, weekly EMA rising, volume spike
            if (close[i] > high_20[i] and ema_rising[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, weekly EMA falling, volume spike
            elif (close[i] < low_20[i] and ema_falling[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 20-day midpoint OR weekly trend turns bearish
            if (close[i] < mid_20[i]) or (~ema_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 20-day midpoint OR weekly trend turns bullish
            if (close[i] > mid_20[i]) or (~ema_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals