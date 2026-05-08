#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA50 trend and volume confirmation
# Uses Donchian channel breakouts for trend capture, filtered by daily EMA50 trend
# and volume spikes to avoid false breakouts. Designed for low-frequency trades
# (<150 total) to minimize fee drag and work in both bull and bear markets.

name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period rolling max/min for Donchian channels
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_1d, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1d, low_min)
    
    # Get 1d data for EMA50 trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike (2.0x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure EMA50 and Donchian have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high with 1d uptrend and volume spike
            if (close[i] > high_max_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low with 1d downtrend and volume spike
            elif (close[i] < low_min_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend fails
            if (close[i] < low_min_aligned[i] or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend fails
            if (close[i] > high_max_aligned[i] or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals