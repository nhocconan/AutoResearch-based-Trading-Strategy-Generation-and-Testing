#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets. 1d EMA34 provides higher-timeframe 
# trend bias to avoid counter-trend trades. Volume confirmation ensures breakouts have 
# participation. Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets via trend-filtered breakouts.

name = "4h_Donchian20_1dEMA34_Volume"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.2 x 20-period EMA
        volume_confirmed = volume[i] > (1.2 * vol_ema_20[i])
        
        # 1d trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1d_aligned[i]
        bearish_trend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + bullish 1d trend
            if (close[i] > highest_high[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + bearish 1d trend
            elif (close[i] < lowest_low[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below midpoint of Donchian channel OR 1d trend turns bearish
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above midpoint of Donchian channel OR 1d trend turns bullish
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals