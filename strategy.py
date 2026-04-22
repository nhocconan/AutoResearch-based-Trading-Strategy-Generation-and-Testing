#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend and volume confirmation.
Long when price breaks above 20-period high with 12-hour EMA rising and volume spike.
Short when price breaks below 20-period low with 12-hour EMA falling and volume spike.
Exit when price crosses the 20-period midpoint. Designed for low trade frequency by requiring
multiple confirmations and using clear price channel structure. Works in both bull and bear
markets by following the 12-hour trend and requiring volume confirmation.
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
    
    # Load 12-hour data for EMA trend - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12-hour EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Donchian channel: 20-period high/low and midpoint
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after enough data for all indicators
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_20[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-period high with 12-hour EMA rising and volume spike
            if (close[i] > high_20[i] and 
                ema34_12h_aligned[i] > ema34_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low with 12-hour EMA falling and volume spike
            elif (close[i] < low_20[i] and 
                  ema34_12h_aligned[i] < ema34_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses the 20-period midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint
                if close[i] < mid_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midpoint
                if close[i] > mid_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_12hEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0