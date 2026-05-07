#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20), price > 12h EMA50, and volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian(20), price < 12h EMA50, and volume > 1.5x 20-period average.
# Exit when price crosses back through the opposite Donchian band or volume drops below average.
# Designed for 4h timeframe with target trade frequency of 25-50/year to minimize fee drag.
# Uses 12h EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
# Works in both bull and bear markets by following the trend on higher timeframe.
name = "4h_Donchian_20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, price > 12h EMA50, volume filter
            long_cond = (close[i] > high_20[i]) and (close[i] > ema50_12h_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below lower Donchian, price < 12h EMA50, volume filter
            short_cond = (close[i] < low_20[i]) and (close[i] < ema50_12h_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower Donchian OR volume filter fails
            if (close[i] < low_20[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper Donchian OR volume filter fails
            if (close[i] > high_20[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals