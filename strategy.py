#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian Breakout with 12-hour EMA Trend and Volume Filter.
Trades breakouts above/below 4-hour Donchian channels only when aligned with 12-hour EMA trend direction.
Requires volume confirmation to reduce false breakouts. Designed for low trade frequency (20-40 trades/year)
to minimize fee drag and capture strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12-hour EMA for trend filter (50-period)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4-hour Donchian channels (20-period)
    # We need to calculate this on 4h data, so we use the prices array directly
    # Since prices is 4h, we can compute rolling max/min
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_filter:
            # Long breakout: price breaks above upper Donchian channel with uptrend bias
            if close[i] > high_max_20[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian channel with downtrend bias
            elif close[i] < low_min_20[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian channel or closes below 12h EMA
                if close[i] < low_min_20[i] or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian channel or closes above 12h EMA
                if close[i] > high_max_20[i] or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0