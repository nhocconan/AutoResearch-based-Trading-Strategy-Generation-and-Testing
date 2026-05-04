#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian breakout captures momentum, EMA50 on 12h filters for trend alignment, volume confirms strength.
# Designed for 20-50 trades/year on 4h to minimize fee drag. Works in bull markets via breakout continuation
# and in bear markets via short breakdowns with trend filter.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirmation"
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
    
    # Get 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe (wait for completed 12h bar)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND 12h EMA50 uptrend
            if (close[i] > highest_high_20[i] and 
                volume_spike[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND 12h EMA50 downtrend
            elif (close[i] < lowest_low_20[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR trend reverses
            if close[i] < lowest_low_20[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR trend reverses
            if close[i] > highest_high_20[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals