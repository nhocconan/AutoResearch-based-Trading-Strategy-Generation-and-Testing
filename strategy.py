#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian breakout captures momentum bursts, EMA50 ensures trend alignment, volume spike validates participation.
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
# Works in bull markets via long breakouts and bear markets via short breakouts with trend filter.

name = "4h_Donchian20_1dEMA50_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        # 1d trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_1d_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + bullish 1d trend
            if (close[i] > highest_high[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume confirmation + bearish 1d trend
            elif (close[i] < lowest_low[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests Donchian middle OR 1d trend turns bearish
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < donchian_middle or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests Donchian middle OR 1d trend turns bullish
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > donchian_middle or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals