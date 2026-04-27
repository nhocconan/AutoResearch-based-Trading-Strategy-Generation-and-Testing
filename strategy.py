#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Uses weekly EMA20 to determine long-term trend: only take long breaks above Donchian(20) high when weekly EMA20 rising
# and short breaks below Donchian(20) low when weekly EMA20 falling. Volume confirmation: current volume > 1.8x 20-period average.
# Designed for 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Donchian(20) on daily data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period EMA on weekly close for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: break above Donchian high AND weekly uptrend AND volume
        if (close[i] > high_roll[i-1] and 
            ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: break below Donchian low AND weekly downtrend AND volume
        elif (close[i] < low_roll[i-1] and 
              ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0