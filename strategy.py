#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian Channel Breakout with 1-week EMA trend filter and volume confirmation.
Trades breakouts above/below 4-hour Donchian(20) channels only when aligned with weekly EMA trend.
Uses volume spike to confirm institutional interest. Designed for low trade frequency (20-50 trades/year)
to minimize fee drag and work in both bull and bear markets by filtering with higher timeframe trend.
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
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x 30-period average
        vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
        if np.isnan(vol_ma_30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above Donchian upper with uptrend bias
            if close[i] > high_max_20[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with downtrend bias
            elif close[i] < low_min_20[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian lower or closes below weekly EMA
                if close[i] < low_min_20[i] or close[i] < ema_34_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above Donchian upper or closes above weekly EMA
                if close[i] > high_max_20[i] or close[i] > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1wEMA34_Volume"
timeframe = "4h"
leverage = 1.0