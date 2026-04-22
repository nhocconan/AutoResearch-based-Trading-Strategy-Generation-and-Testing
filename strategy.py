#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian breakout with 1-week trend filter and volume confirmation.
Trades breakouts above/below Donchian(20) channels in the direction of weekly EMA(50).
Uses volume spike (>2x 20-period average) to confirm institutional interest.
Designed for low trade frequency (~20-50 trades/year) to minimize fee drag.
Works in both bull and bear markets by following higher timeframe trend.
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (50-period)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above upper Donchian with uptrend bias
            if close[i] > high_max_20[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with downtrend bias
            elif close[i] < low_min_20[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below lower Donchian or weekly EMA
                if close[i] < low_min_20[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price closes above upper Donchian or weekly EMA
                if close[i] > high_max_20[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1wEMA50_Volume"
timeframe = "4h"
leverage = 1.0