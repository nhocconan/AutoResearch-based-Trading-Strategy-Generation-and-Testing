#!/usr/bin/env python3
name = "1d_PremiumBreakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly context: 1w trend via EMA34
    df_1w = get_htf_data(prices, '1w')
    close_w = df_1w['close'].values
    ema34_w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_w_aligned = align_htf_to_ltf(prices, df_1w, ema34_w)
    
    # Daily indicators
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # wait for EMA warmup
        # Skip if weekly trend not ready
        if np.isnan(ema34_w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        weekly_up = close[i] > ema34_w_aligned[i]
        weekly_down = close[i] < ema34_w_aligned[i]
        
        if position == 0:
            # Long: weekly uptrend + breakout above Donchian high + volume spike
            if (weekly_up and 
                close[i] > donch_high[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + breakdown below Donchian low + volume spike
            elif (weekly_down and 
                  close[i] < donch_low[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low or trend change
            if close[i] < donch_low[i] or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above Donchian high or trend change
            if close[i] > donch_high[i] or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals