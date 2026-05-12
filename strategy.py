#!/usr/bin/env python3
name = "4h_Donchian20_VolumeTrend_12hTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    trend_12h_up = close_12h > ema_50_12h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up)
    
    # Donchian(20) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure EMA and Donchian have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 12h uptrend + price breaks above Donchian high + volume spike
            if (trend_12h_up_aligned[i] and 
                close[i] > highest_20[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h downtrend + price breaks below Donchian low + volume spike
            elif ((~trend_12h_up_aligned[i]) and 
                  close[i] < lowest_20[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 12h trend turns down
            if close[i] < lowest_20[i] or (~trend_12h_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 12h trend turns up
            if close[i] > highest_20[i] or trend_12h_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals