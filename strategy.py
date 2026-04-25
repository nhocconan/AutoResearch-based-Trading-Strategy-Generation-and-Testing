#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Donchian channel breakouts on 12h timeframe capture medium-term trends.
Using 1d EMA50 for trend filter ensures alignment with daily momentum.
Volume confirmation avoids false breakouts. Works in bull/bear by following 1d trend.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 12h Donchian(20) from previous completed 12h bars
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need 20 for Donchian + 1 for previous bar
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    # Upper/lower channel from previous 20 completed 12h bars
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().shift(1).values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need 50 for EMA + 1 for previous bar
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + EMA50 + VolMA20
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_50_level = ema_50_1d_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Breakout conditions
        bullish_breakout = curr_close > upper_12h_aligned[i]  # Break above upper Donchian
        bearish_breakout = curr_close < lower_12h_aligned[i]  # Break below lower Donchian
        
        # Exit conditions: reverse breakout or trend change
        if position != 0:
            if position == 1 and (curr_close < lower_12h_aligned[i] or curr_close < ema_50_level):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and (curr_close > upper_12h_aligned[i] or curr_close > ema_50_level):
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Breakout + trend + volume
        if position == 0:
            long_condition = bullish_breakout and (curr_close > ema_50_level) and volume_spike
            short_condition = bearish_breakout and (curr_close < ema_50_level) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0