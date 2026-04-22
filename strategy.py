#!/usr/bin/env python3
"""
Hypothesis: 12-hour Ehlers Fisher Transform with 1-day trend filter and volume confirmation.
Long when Fisher crosses above -1.5 with 1-day EMA50 uptrend and volume > 1.5x average.
Short when Fisher crosses below +1.5 with 1-day EMA50 downtrend and volume > 1.5x average.
Exit when Fisher crosses zero.
Fisher Transform identifies turning points in price with Gaussian distribution, effective in
both trending and ranging markets. Combined with trend filter and volume confirmation,
this reduces false signals and limits trade frequency (~10-20/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for trend filter and volume calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ehlers Fisher Transform (9-period)
    # Normalize price to [-1, 1] range over lookback period
    def fish_transform(price_series, length=9):
        if len(price_series) < length:
            return np.full_like(price_series, np.nan)
        
        highest = np.max(price_series)
        lowest = np.min(price_series)
        if highest == lowest:
            return np.zeros_like(price_series)
        
        # Normalize to [-1, 1]
        value = 2 * ((price_series - lowest) / (highest - lowest) - 0.5)
        # Clamp to avoid domain error in log
        value = np.clip(value, -0.999, 0.999)
        # Fisher transform
        fish = 0.5 * np.log((1 + value) / (1 - value))
        # Smooth with 2-period EMA
        fish_smoothed = pd.Series(fish).ewm(span=2, adjust=False).mean().values
        return fish_smoothed
    
    # Calculate Fisher Transform
    fish = fish_transform(close, 9)
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: 20-period average volume on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if data not ready
        if (np.isnan(fish[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 with uptrend and volume confirmation
            if fish[i] > -1.5 and fish[i-1] <= -1.5 and close[i] > ema50_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 with downtrend and volume confirmation
            elif fish[i] < 1.5 and fish[i-1] >= 1.5 and close[i] < ema50_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Fisher crosses zero (or crosses below -1.5 for faster exit)
                if fish[i] < 0 and fish[i-1] >= 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Fisher crosses zero (or crosses above +1.5 for faster exit)
                if fish[i] > 0 and fish[i-1] <= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_EhlersFisher_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0