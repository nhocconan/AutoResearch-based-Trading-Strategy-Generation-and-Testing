#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeFilter_v1
Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation. 
In bull markets: long when price breaks above Donchian upper channel with 1w uptrend and volume > 1.5x average.
In bear markets: short when price breaks below Donchian lower channel with 1w downtrend and volume > 1.5x average.
Uses discrete position sizing (0.25) to minimize fee drag. Targets 12-37 trades/year (50-150 over 4 years) by requiring confluence of breakout, trend, and volume.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Donchian Channel (20-period)
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    htf_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 50 for EMA, 20 for volume MA)
    start_idx = max(donchian_period, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > upper[i-1] and htf_trend[i] == 1 and volume_confirm
        bearish_breakout = close[i] < lower[i-1] and htf_trend[i] == -1 and volume_confirm
        
        # Exit conditions: reverse breakout or loss of trend alignment
        bullish_exit = position == 1 and (close[i] < lower[i] or htf_trend[i] == -1)
        bearish_exit = position == -1 and (close[i] > upper[i] or htf_trend[i] == 1)
        
        if bullish_breakout and position != 1:
            signals[i] = 0.25
            position = 1
        elif bearish_breakout and position != -1:
            signals[i] = -0.25
            position = -1
        elif bullish_exit:
            signals[i] = 0.0
            position = 0
        elif bearish_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0