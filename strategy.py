#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above 20-period high with weekly pivot bullish and volume > 1.8x average
# Short when price breaks below 20-period low with weekly pivot bearish and volume > 1.8x average
# Exit when price retraces to 10-period EMA or reverses to opposite Donchian level
# Uses Donchian for breakout structure, weekly pivot for market regime, volume for conviction
# Designed to work in both bull and bear markets by filtering breakouts with weekly trend
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_Donchian_20_WeeklyPivot_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high)
    low_roll = pd.Series(low)
    donchian_high = high_roll.rolling(window=20, min_periods=20).max().values
    donchian_low = low_roll.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate weekly pivot point (using weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_week_high = df_1w['high'].shift(1)
    prev_week_low = df_1w['low'].shift(1)
    prev_week_close = df_1w['close'].shift(1)
    
    # Calculate weekly pivot point
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    # Weekly pivot bullish if close > pivot, bearish if close < pivot
    weekly_bullish = prev_week_close > weekly_pivot
    weekly_bearish = prev_week_close < weekly_pivot
    
    # Align weekly pivot and bias to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.values.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.values.astype(float))
    
    # Calculate 10-period EMA for exit
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(ema_10[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, weekly bullish, volume spike
            if (close[i] > donchian_high[i] and 
                weekly_bullish_aligned[i] > 0.5 and  # Weekly pivot bullish
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, weekly bearish, volume spike
            elif (close[i] < donchian_low[i] and 
                  weekly_bearish_aligned[i] > 0.5 and  # Weekly pivot bearish
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to 10 EMA or reverses to Donchian low
            if (close[i] <= ema_10[i]) or (close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to 10 EMA or reverses to Donchian high
            if (close[i] >= ema_10[i]) or (close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals