#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Primary indicators on 4h
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # 4h volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # 12h trend filter - EMA34
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = np.full(len(close_12h), np.nan)
    for i in range(34, len(close_12h)):
        ema_34_12h[i] = np.mean(close_12h[i-33:i+1]) if i == 34 else (ema_34_12h[i-1] * 33 + close_12h[i]) / 34
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup period
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        long_breakout = close[i] > highest_high[i] and volume[i] > 1.5 * vol_ma[i]
        short_breakout = close[i] < lowest_low[i] and volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: only trade in direction of 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume spike
            if long_breakout and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume spike
            elif short_breakout and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reverses
            if close[i] < lowest_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reverses
            if close[i] > highest_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0