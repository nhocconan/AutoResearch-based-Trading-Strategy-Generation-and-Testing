#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: Donchian channel breakout (20-day) on daily timeframe filtered by weekly trend (EMA 50) and volume confirmation.
Goes long when price breaks above upper Donchian band with volume > 20-day average and weekly EMA50 trending up.
Goes short when price breaks below lower Donchian band with volume > 20-day average and weekly EMA50 trending down.
Designed for low-frequency trading (~10-20 trades/year) to minimize fee impact and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_roll[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_roll[i-1]  # Break below previous period's low
        
        # Weekly trend filter
        if i >= 1:
            weekly_up = ema50_1w_aligned[i] > ema50_1w_aligned[i-1]
            weekly_down = ema50_1w_aligned[i] < ema50_1w_aligned[i-1]
        else:
            weekly_up = False
            weekly_down = False
        
        if position == 1:  # Long position
            # Exit: breakdown below lower Donchian band or weekly trend turns down
            if close[i] < low_roll[i] or not weekly_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: breakout above upper Donchian band or weekly trend turns up
            if close[i] > high_roll[i] or not weekly_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Donchian breakout up with volume confirmation and weekly trend up
            if breakout_up and vol_confirmed and weekly_up:
                position = 1
                signals[i] = 0.25
            # Short: Donchian breakout down with volume confirmation and weekly trend down
            elif breakout_down and vol_confirmed and weekly_down:
                position = -1
                signals[i] = -0.25
    
    return signals