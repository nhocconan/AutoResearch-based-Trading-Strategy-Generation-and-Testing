#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation.
Breakouts capture momentum, EMA34 filters direction, volume ensures conviction.
Designed for 20-30 trades/year to minimize fee drag while capturing momentum bursts.
Works in bull markets (buy upper Donchian breakout in uptrend) and bear markets (sell lower Donchian breakout in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_len = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(donchian_len - 1, n):
        donchian_upper[i] = np.max(high[i - donchian_len + 1:i + 1])
        donchian_lower[i] = np.min(low[i - donchian_len + 1:i + 1])
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2/35) + (ema_34_1d[i-1] * 33/35)
    
    # Align 1d EMA to 4h timeframe
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_len, 20, 34)  # need Donchian, volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_4h[i]
        trend_down = close[i] < ema_34_1d_4h[i]
        
        if position == 0:
            # Long entry: close above upper Donchian with volume and uptrend
            if (close[i] > donchian_upper[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below lower Donchian with volume and downtrend
            elif (close[i] < donchian_lower[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below lower Donchian or reverse signal
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above upper Donchian or reverse signal
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0