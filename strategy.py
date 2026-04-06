#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA(50) trend filter and volume confirmation
# Enter long when price breaks above 20-day high + price > weekly EMA(50) + volume > 2x average
# Enter short when price breaks below 20-day low + price < weekly EMA(50) + volume > 2x average
# Exit when price crosses 10-day moving average in opposite direction
# Uses weekly trend to avoid counter-trend trades, targeting 60-100 trades over 4 years

name = "1d_donchian20_weeklyema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on daily
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # Exit condition: 10-day EMA crossover
    ema_10 = pd.Series(close).ewm(span=10, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(ema_10[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below 10-day EMA
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above 10-day EMA
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > ema_50_aligned[i]:
                    # Breakout above 20-day high with weekly uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and close[i] < ema_50_aligned[i]:
                    # Breakdown below 20-day low with weekly downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals