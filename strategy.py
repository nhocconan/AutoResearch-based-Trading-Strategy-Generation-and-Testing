#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# Long when: price > Donchian Upper(20), price > 1d EMA(50), volume > 1.5x 20-period average
# Short when: price < Donchian Lower(20), price < 1d EMA(50), volume > 1.5x 20-period average
# Exit when: price crosses back below/above Donchian midpoint OR 1d EMA trend reverses
# Position size: 0.28 (28% of capital) to balance return and drawdown. Target: 15-30 trades/year.
# Designed to work in both bull (breakout with trend) and bear (breakdown with trend) markets.

name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper = high_20.values
    lower = low_20.values
    midpoint = (upper + lower) / 2
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Upper(20), price > EMA50, volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_50_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: price < Lower(20), price < EMA50, volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price < midpoint OR EMA trend turns bearish (price < EMA50)
            if (close[i] < midpoint[i]) or (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price > midpoint OR EMA trend turns bullish (price > EMA50)
            if (close[i] > midpoint[i]) or (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals