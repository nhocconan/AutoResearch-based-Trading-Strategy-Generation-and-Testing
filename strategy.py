#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day trend filter and volume confirmation.
Long when price breaks above 10-period high with 1-day EMA34 rising and volume spike.
Short when price breaks below 10-period low with 1-day EMA34 falling and volume spike.
Exit when price crosses 10-period moving average.
Designed for low trade frequency by requiring multiple confirmations and using 12h timeframe.
Works in both bull and bear markets by following the daily trend.
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
    
    # Load 1-day data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channel: 10-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low = low_series.rolling(window=10, min_periods=10).min().values
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(ma_10[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with 1-day EMA34 rising and volume spike
            if (close[i] > donchian_high[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with 1-day EMA34 falling and volume spike
            elif (close[i] < donchian_low[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 10-period moving average
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below MA
                if close[i] < ma_10[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above MA
                if close[i] > ma_10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0