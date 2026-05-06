#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly EMA(200) trend filter + volume confirmation
# Uses weekly trend direction to filter breakouts: only long when price > weekly EMA200, short when price < weekly EMA200
# Donchian(20) breakout provides clear entry/exit signals with defined risk
# Volume confirmation (>1.5x 20-bar average) ensures institutional participation
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Weekly EMA200 filter avoids counter-trend trades in strong trends, reducing whipsaws
# Works in bull markets (catching uptrend breakouts) and bear markets (catching downtrend breakdowns)

name = "6h_Donchian20_WeeklyEMA200_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA(200)
    ema_200_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Donchian(20) channels on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align weekly EMA200 to 6h timeframe
    ema_200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_200_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_200_weekly_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND above weekly EMA200 AND volume confirmation
            if (close[i] > high_20[i] and close[i] > ema_200_weekly_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below weekly EMA200 AND volume confirmation
            elif (close[i] < low_20[i] and close[i] < ema_200_weekly_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals