#!/usr/bin/env python3
"""
Hypothesis: 6-hour Elder Ray Index with 1-day trend filter and volume confirmation.
Long when Bull Power > 0 (close > EMA13) with Bear Power improving and volume > 1.5x average.
Short when Bear Power < 0 (close < EMA13) with Bull Power deteriorating and volume > 1.5x average.
Exit when power crosses zero or volume drops. Uses 1-day EMA34 for trend filter to avoid counter-trend trades.
Designed for low trade frequency (~15-25/year) to avoid fee drag in 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load 1-day data for trend filter and volume average - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1-day average volume for filter
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power improving (less negative) AND volume confirmation AND uptrend
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and 
                volume[i] > 1.5 * vol_avg_1d_aligned[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power deteriorating (less positive) AND volume confirmation AND downtrend
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and 
                  volume[i] > 1.5 * vol_avg_1d_aligned[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 OR Bear Power deteriorating OR volume drops OR trend change
                if (bull_power[i] <= 0 or 
                    bear_power[i] < bear_power[i-1] or
                    volume[i] < vol_avg_1d_aligned[i] or
                    close[i] < ema34_1d_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bear Power >= 0 OR Bull Power deteriorating OR volume drops OR trend change
                if (bear_power[i] >= 0 or 
                    bull_power[i] < bull_power[i-1] or
                    volume[i] < vol_avg_1d_aligned[i] or
                    close[i] > ema34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0