#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily Donchian(20) from previous day's close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_shift = pd.Series(high_1d).shift(1).values
    low_1d_shift = pd.Series(low_1d).shift(1).values
    
    upper_20 = pd.Series(high_1d_shift).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d_shift).rolling(window=20, min_periods=20).min().values
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1d_aligned[i]
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + uptrend + volume spike
            if (close[i] > upper_val and 
                close[i] > ema_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + downtrend + volume spike
            elif (close[i] < lower_val and 
                  close[i] < ema_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR trend turns down
            if (close[i] < lower_val or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR trend turns up
            if (close[i] > upper_val or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals