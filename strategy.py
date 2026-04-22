#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA200 trend filter and volume confirmation.
Long when price breaks above 20-period high with 1-day EMA200 rising and volume > 1.5x average.
Short when price breaks below 20-period low with 1-day EMA200 falling and volume > 1.5x average.
Exit when price crosses the 20-period moving average.
Designed for low trade frequency by requiring multiple confirmations and using daily trend filter.
Works in both bull and bear markets by following the higher timeframe trend.
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
    
    # Load 1-day data for EMA200 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Donchian channels (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period moving average for exit
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Donchian
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(ma20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-period high with 1-day EMA200 rising and volume confirmation
            if (close[i] > high_roll[i] and 
                ema200_1d_aligned[i] > ema200_1d_aligned[i-1] and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low with 1-day EMA200 falling and volume confirmation
            elif (close[i] < low_roll[i] and 
                  ema200_1d_aligned[i] < ema200_1d_aligned[i-1] and vol_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 20-period moving average
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below MA20
                if close[i] < ma20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above MA20
                if close[i] > ma20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dEMA200_Trend_Volume"
timeframe = "4h"
leverage = 1.0