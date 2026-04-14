#!/usr/bin/env python3
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
    
    # Load 1d data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(high_1d, 1)), 
                               np.abs(low_1d - np.roll(low_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first period
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band in uptrend (price > EMA50)
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1d_aligned[i] and
                atr_14_1d_aligned[i] > 0):  # Ensure volatility present
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower band in downtrend (price < EMA50)
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  atr_14_1d_aligned[i] > 0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian middle or trend reverses
            mid_20 = (high_20[i] + low_20[i]) / 2
            if close[i] < mid_20 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian middle or trend reverses
            mid_20 = (high_20[i] + low_20[i]) / 2
            if close[i] > mid_20 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_EMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0