#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian channel provides clear structure for breakouts in both bull and bear markets.
# 12h EMA50 offers higher-timeframe trend bias to avoid counter-trend trades.
# Volume confirmation ensures breakouts have participation.
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets via trend-filtered breakouts.

name = "4h_Donchian20_12hEMA50_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        # 12h trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_12h_aligned[i]
        bearish_trend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Close breaks above upper Donchian + volume confirmation + bullish 12h trend
            if (close[i] > highest_high[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian + volume confirmation + bearish 12h trend
            elif (close[i] < lowest_low[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close drops below lower Donchian OR 12h trend turns bearish
            if (close[i] < lowest_low[i] or bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises above upper Donchian OR 12h trend turns bullish
            if (close[i] > highest_high[i] or bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals