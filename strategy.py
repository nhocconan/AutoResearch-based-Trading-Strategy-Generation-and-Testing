#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend and volume confirmation.
Long when price breaks above Donchian(20) upper band with 12h EMA50 rising and volume spike.
Short when price breaks below Donchian(20) lower band with 12h EMA50 falling and volume spike.
Exit when price returns to the 12h EMA50. Uses 12h timeframe for trend to reduce whipsaw,
volume for confirmation, and EMA for exit to capture trends. Designed for low trade frequency
by requiring multiple confirmations. Works in both bull and bear markets by following the
12h trend.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period) on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for EMA50 and Donchian
        # Skip if data not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with 12h EMA50 rising and volume spike
            if (close[i] > high_max_20[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with 12h EMA50 falling and volume spike
            elif (close[i] < low_min_20[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to 12h EMA50 (trend exhaustion signal)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 12h EMA50
                if close[i] < ema50_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 12h EMA50
                if close[i] > ema50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0