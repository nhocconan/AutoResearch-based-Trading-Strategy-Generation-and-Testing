#!/usr/bin/env python3
"""
4h_Donchian_Breakout_12h_EMA_Trend_Volume
Hypothesis: Donchian breakout with 12h EMA trend filter and volume confirmation.
Works in bull markets via breakout momentum and in bear via shorting breakdowns.
Volume ensures momentum validity. 12h EMA provides stable trend filter.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # 12h EMA trend filter (34-period)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = np.full(len(close_12h), np.nan)
    k = 2 / (34 + 1)
    for i in range(34, len(close_12h)):
        if i == 34:
            ema_34_12h[i] = np.mean(close_12h[i-34:i+1])
        else:
            ema_34_12h[i] = close_12h[i] * k + ema_34_12h[i-1] * (1 - k)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with uptrend and volume
            if close[i] > donchian_high[i] and close[i] > ema_34_12h_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with downtrend and volume
            elif close[i] < donchian_low[i] and close[i] < ema_34_12h_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below Donchian lower or trend flips
            if close[i] < donchian_low[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Donchian upper or trend flips
            if close[i] > donchian_high[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12h_EMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0