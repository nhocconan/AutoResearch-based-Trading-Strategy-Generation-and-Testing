#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA21 trend and volume spike
# Uses Donchian breakouts for trend capture, filtered by daily EMA21 trend and volume spikes.
# Designed for low-frequency trades (target 50-150 total) to minimize fee drift.
# Works in both bull and bear markets by following the daily trend direction.

name = "12h_Donchian20_1dEMA21_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA21
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Highest high and lowest low over last 20 periods
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    for i in range(20, n):
        high_max[i] = np.max(high[i-20:i])
        low_min[i] = np.min(low[i-20:i])
    
    # Volume spike (2.0x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high with 1d uptrend and volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema21_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low with 1d downtrend and volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema21_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend fails
            if (close[i] < low_min[i] or 
                close[i] < ema21_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend fails
            if (close[i] > high_max[i] or 
                close[i] > ema21_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals