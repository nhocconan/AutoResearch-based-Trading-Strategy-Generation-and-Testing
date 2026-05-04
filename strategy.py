#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian channels capture institutional breakouts. EMA50 on 1d ensures medium-term trend alignment.
# Volume spike confirms institutional participation. Designed for 12-37 trades/year on 12h to minimize fee drag.
# Works in bull markets via long upper band breaks in uptrend and bear markets via lower band breaks in downtrend.

name = "12h_Donchian20_VolumeSpike_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 12h timeframe
    # Upper band = highest high over past 20 periods
    # Lower band = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND 1d uptrend AND volume spike
            if (close[i] > upper_band[i] and 
                close[i] > ema_50_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND 1d downtrend AND volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band OR 1d trend turns down
            if close[i] < lower_band[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band OR 1d trend turns up
            if close[i] > upper_band[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals