#!/usr/bin/env python3
"""
4h_Donchian20_12hEMA_Trend
Hypothesis: Use 4h Donchian(20) breakouts aligned with 12h EMA(34) trend direction, confirmed by volume > 1.8x 24-period average. The 12h EMA provides a smoother trend filter than 4h indicators, reducing whipsaw in choppy markets. Enter long when price breaks above upper Donchian channel and 12h EMA is rising; short when price breaks below lower Donchian channel and 12h EMA is falling. This combination captures strong trends while avoiding false breakouts in sideways markets. Designed to work in both bull and bear markets by following the dominant trend on 12h timeframe. Targets 20-40 trades/year via strict breakout conditions.
"""

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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(34)
    ema_period = 34
    ema_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period-1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (ema_period + 1)) + (ema_12h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h Donchian channels (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n >= donch_len:
        for i in range(donch_len-1, n):
            upper[i] = np.max(high[i-donch_len+1:i+1])
            lower[i] = np.min(low[i-donch_len+1:i+1])
    
    # Volume confirmation: volume > 1.8x 24-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    if n >= vol_period:
        for i in range(vol_period, n):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_len, vol_period, 35)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: break above upper Donchian + rising 12h EMA + volume
            if close[i] > upper[i] and i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] > ema_12h_aligned[i-1] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + falling 12h EMA + volume
            elif close[i] < lower[i] and i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] < ema_12h_aligned[i-1] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below lower Donchian or 12h EMA turns down
            if close[i] < lower[i] or (i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above upper Donchian or 12h EMA turns up
            if close[i] > upper[i] or (i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA_Trend"
timeframe = "4h"
leverage = 1.0