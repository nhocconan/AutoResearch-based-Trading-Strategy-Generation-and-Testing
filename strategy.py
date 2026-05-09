#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA filter and volume confirmation
# Long when: price breaks above Donchian(20) upper band, price > 1d EMA(50), volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) lower band, price < 1d EMA(50), volume > 1.5x 20-period average
# Exit when: price crosses back below/above Donchian(20) middle band or EMA(50) flips direction
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 25-50 trades/year.
# Designed to work in both bull (breakouts with trend) and bear (breakouts against trend) markets.

name = "4h_Donchian20_1dEMA_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    mid_20 = (high_20 + low_20) / 2
    
    # Get 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, price > EMA(50), volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, price < EMA(50), volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle band or EMA(50) turns down
            if (close[i] < mid_20[i]) or (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle band or EMA(50) turns up
            if (close[i] > mid_20[i]) or (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals